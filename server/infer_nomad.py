#!/usr/bin/env python3
"""
NoMaD inference — standalone, no ROS.

Two modes controlled by --mode:
  navigate   Goal-conditioned navigation: provide --goal image.
             Outputs distance estimate + 8 cumulative waypoints.
  explore    Exploration: no goal needed.
             Goal token is masked; model outputs exploratory trajectory.

Image spec: 96×96 square, ImageNet-normalised.
The input frames are center-cropped to square then resized.
"""
import argparse, os, sys, warnings
import numpy as np
import torch, yaml
from PIL import Image as PILImage
from typing import List, Tuple

_VISUALNAV = os.environ.get("VISUALNAV", os.path.expanduser("~/visualnav-transformer"))
sys.path.insert(0, os.path.join(_VISUALNAV, "train"))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# Normalisation stats for the diffusion output (delta x, delta y)
# Source: train/vint_train/data/data_config.yaml
ACTION_STATS = {
    "min": np.array([-2.5, -4.0], dtype=np.float32),
    "max": np.array([ 5.0,  4.0], dtype=np.float32),
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def center_crop_square(img: PILImage.Image) -> PILImage.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def transform_images(pil_imgs: List[PILImage.Image], size: int) -> torch.Tensor:
    tensors = []
    for img in pil_imgs:
        img = center_crop_square(img).resize((size, size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0))
    return torch.cat(tensors, dim=1)  # [1, 3*N, size, size]


def load_images_from_dir(img_dir: str) -> Tuple[List[PILImage.Image], List[str]]:
    exts = {".jpg", ".jpeg", ".png"}
    def sort_key(fname):
        stem = os.path.splitext(fname)[0]
        try: return (0, int(stem))
        except ValueError: return (1, stem)
    names = sorted(
        (p for p in os.listdir(img_dir) if os.path.splitext(p)[1].lower() in exts),
        key=sort_key,
    )
    if not names:
        raise FileNotFoundError(f"No images in {img_dir}")
    return [PILImage.open(os.path.join(img_dir, n)).convert("RGB") for n in names], names


def load_nomad(ckpt_path: str, config: dict, device: torch.device) -> NoMaD:
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)

    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    dist_pred_net = DenseNetwork(embedding_dim=config["encoding_size"])

    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_net,
    )

    # NoMaD checkpoint is a plain state dict (unlike ViNT which wraps it)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def unnormalize_and_cumsum(ndeltas: np.ndarray) -> np.ndarray:
    """Map normalised deltas in [-1,1] → metres, then cumsum to positions."""
    ndeltas = (ndeltas + 1) / 2
    deltas  = ndeltas * (ACTION_STATS["max"] - ACTION_STATS["min"]) + ACTION_STATS["min"]
    return np.cumsum(deltas, axis=1)   # [B, T, 2] cumulative positions


def run_diffusion(model, encoding, num_diffusion_iters, len_traj_pred,
                  num_samples, device):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    # Repeat encoding for each sample
    cond = encoding.repeat(num_samples, 1)

    naction = torch.randn((num_samples, len_traj_pred, 2), device=device)
    noise_scheduler.set_timesteps(num_diffusion_iters)

    for k in noise_scheduler.timesteps:
        noise_pred = model("noise_pred_net",
                           sample=naction, timestep=k, global_cond=cond)
        naction = noise_scheduler.step(noise_pred, k, naction).prev_sample

    return naction  # [num_samples, T, 2] — still normalised deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obs",  required=True, help="Directory of observation frames")
    parser.add_argument("--goal", default=None,  help="Goal image (navigate mode only)")
    parser.add_argument("--mode", default="navigate", choices=["navigate", "explore"],
                        help="navigate = goal-conditioned, explore = no goal")
    parser.add_argument("--ckpt", default=os.path.join(_VISUALNAV,
                        "deployment", "model_weights", "nomad.pth"))
    parser.add_argument("--config", default=os.path.join(_VISUALNAV,
                        "train", "config", "nomad.yaml"))
    parser.add_argument("--waypoint",    type=int, default=2,
                        help="Waypoint index to act on (0-based, default 2)")
    parser.add_argument("--num-samples", type=int, default=4,
                        help="Number of diffusion trajectory samples (default 4)")
    args = parser.parse_args()

    if args.mode == "navigate" and args.goal is None:
        parser.error("--goal is required for navigate mode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    image_size    = config["image_size"][0]   # 96 (square)
    context_size  = config["context_size"]    # 3
    len_traj_pred = config["len_traj_pred"]   # 8
    num_diff_iters = config["num_diffusion_iters"]  # 10
    max_v         = 0.2   # m/s
    frame_rate    = 4     # Hz

    print(f"Loading NoMaD from {args.ckpt} ...")
    model = load_nomad(args.ckpt, config, device)
    print("Model ready.\n")

    all_imgs, all_names = load_images_from_dir(args.obs)
    obs_imgs  = all_imgs[-(context_size + 1):]
    obs_names = all_names[-(context_size + 1):]

    obs_tensor = transform_images(obs_imgs, image_size).to(device)

    if args.mode == "navigate":
        goal_img    = PILImage.open(args.goal).convert("RGB")
        goal_tensor = transform_images([goal_img], image_size).to(device)
        goal_mask   = torch.zeros(1, dtype=torch.long, device=device)  # 0 = use goal
    else:
        goal_tensor = torch.randn(1, 3, image_size, image_size, device=device)
        goal_mask   = torch.ones(1, dtype=torch.long, device=device)   # 1 = mask goal

    print(f"Observation frames ({len(obs_imgs)}):")
    for n in obs_names:
        print(f"  {n}")
    if args.mode == "navigate":
        print(f"Goal image: {args.goal}")
    else:
        print("Mode: EXPLORATION (no goal)")

    with torch.no_grad():
        # 1. Encode observations + goal
        encoding = model("vision_encoder",
                         obs_img=obs_tensor,
                         goal_img=goal_tensor,
                         input_goal_mask=goal_mask)   # [1, 256]

        # 2. Distance prediction
        dist = model("dist_pred_net", obsgoal_cond=encoding).item()

        # 3. Diffusion: generate trajectory samples
        naction = run_diffusion(model, encoding, num_diff_iters,
                                len_traj_pred, args.num_samples, device)

    # Un-normalise deltas and accumulate into cumulative positions.
    # ACTION_STATS are in the training data's normalised unit; multiply by
    # max_v / frame_rate to convert to actual metres per timestep.
    naction_np = naction.cpu().numpy()                   # [B, T, 2]
    waypoints_all = unnormalize_and_cumsum(naction_np)   # [B, T, 2]
    if config["normalize"]:
        waypoints_all *= (max_v / frame_rate)            # → metres

    print()
    print("=" * 48)
    print(f"  Temporal distance to goal: {dist:6.2f} steps")
    print(f"  (range 0–20; <3 = effectively at goal)")
    print("=" * 48)

    chosen_wp = args.waypoint
    print(f"\n  {args.num_samples} sampled trajectories (waypoints are cumulative positions):")
    for s, wps in enumerate(waypoints_all):
        marker = " ◀ chosen sample" if s == 0 else ""
        chosen = wps[chosen_wp]
        print(f"  Sample {s}: wp[{chosen_wp}] → x={chosen[0]:.4f}m  y={chosen[1]:.4f}m{marker}")

    print()
    print(f"  Full chosen trajectory (sample 0):")
    print(f"  {'idx':>4}  {'x (m)':>8}  {'y (m)':>8}")
    for i, (x, y) in enumerate(waypoints_all[0]):
        marker = " ◀" if i == chosen_wp else ""
        print(f"  [{i}]  {x:8.4f}  {y:8.4f}{marker}")
    print("=" * 48)


if __name__ == "__main__":
    main()
