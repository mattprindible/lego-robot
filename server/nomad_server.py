#!/usr/bin/env python3
"""
NoMaD inference server — runs on Jetson Orin Nano.

Receives JPEG frames from the iPhone app via WebSocket, runs NoMaD
inference on GPU, and sends drive commands back through the iPhone's
BLE bridge to the LEGO Inventor Hub.

Usage (nomad venv on Jetson):
  python nomad_server.py                            # explore mode
  python nomad_server.py --goal /path/to/goal.jpg  # navigate mode

Then connect the iPhone app to:  ws://<jetson-ip>:8765
"""
import argparse, asyncio, math, os, sys, warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import torch, yaml
from PIL import Image as PILImage

# ── visualnav-transformer location ───────────────────────────────────────────
# Set VISUALNAV env var to override, e.g.:  export VISUALNAV=/opt/visualnav-transformer
_VISUALNAV = os.environ.get("VISUALNAV", os.path.expanduser("~/visualnav-transformer"))
sys.path.insert(0, os.path.join(_VISUALNAV, "train"))

from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from sensor_hub import SensorHub, local_ip

# ── normalisation constants (from train/vint_train/data/data_config.yaml) ────
ACTION_STATS = {
    "min": np.array([-2.5, -4.0], dtype=np.float32),
    "max": np.array([ 5.0,  4.0], dtype=np.float32),
}
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_DEFAULT_CKPT      = os.path.join(_VISUALNAV, "deployment", "model_weights", "nomad.pth")
_DEFAULT_CFG       = os.path.join(_VISUALNAV, "train", "config", "nomad.yaml")
_DEFAULT_ROBOT_CFG = os.path.join(os.path.dirname(__file__), "..", "config", "robot.yaml")


# ── image preprocessing ───────────────────────────────────────────────────────
def _center_crop_square(img: PILImage.Image) -> PILImage.Image:
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _transform(pil_imgs: list, size: int) -> torch.Tensor:
    tensors = []
    for img in pil_imgs:
        img = _center_crop_square(img).resize((size, size))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        tensors.append(torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0))
    return torch.cat(tensors, dim=1)  # [1, 3*N, size, size]


# ── model loading ─────────────────────────────────────────────────────────────
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


# ── inference (runs in thread executor) ──────────────────────────────────────
def _run_inference(obs_imgs: list, goal_tensor: torch.Tensor,
                   goal_mask: torch.Tensor, model: NoMaD,
                   config: dict, device: torch.device,
                   max_v: float, frame_rate: float):
    """Blocking inference call. Returns (waypoints [T,2] in metres, dist float)."""
    image_size     = config["image_size"][0]
    len_traj_pred  = config["len_traj_pred"]
    num_diff_iters = config["num_diffusion_iters"]

    obs_tensor = _transform(obs_imgs, image_size).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diff_iters,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    with torch.no_grad():
        encoding = model("vision_encoder",
                         obs_img=obs_tensor,
                         goal_img=goal_tensor,
                         input_goal_mask=goal_mask)  # [1, encoding_size]
        dist = model("dist_pred_net", obsgoal_cond=encoding).item()

        # Diffusion: single sample per inference step
        cond    = encoding  # already [1, encoding_size]
        naction = torch.randn((1, len_traj_pred, 2), device=device)
        noise_scheduler.set_timesteps(num_diff_iters)
        for k in noise_scheduler.timesteps:
            noise_pred = model("noise_pred_net",
                               sample=naction, timestep=k, global_cond=cond)
            naction = noise_scheduler.step(noise_pred, k, naction).prev_sample

    naction_np = naction.cpu().numpy()[0]           # [T, 2]
    naction_np = (naction_np + 1) / 2
    deltas     = (naction_np * (ACTION_STATS["max"] - ACTION_STATS["min"])
                  + ACTION_STATS["min"])
    waypoints  = np.cumsum(deltas, axis=0)          # cumulative positions, training units
    if config["normalize"]:
        waypoints *= (max_v / frame_rate)           # → metres

    return waypoints, dist


# ── waypoint → Pybricks drive command ─────────────────────────────────────────
def _wp_to_drive(wp_x: float, wp_y: float, chosen_wp: int,
                 max_v: float, max_w: float, frame_rate: float) -> str:
    """
    Convert a NoMaD waypoint to a Pybricks DriveBase command string.

    wp_x = forward distance (m), wp_y = left distance (m) in robot frame.
    chosen_wp = the waypoint index, which tells us how many timesteps away it is.

    Returns 'drive:<speed_mm>:<turn_deg>' where speed is in mm/s and turn in deg/s.
    """
    T = (chosen_wp + 1) / frame_rate           # seconds until we should reach wp

    dist    = math.sqrt(wp_x ** 2 + wp_y ** 2)
    v       = min(max_v, max(0.0, dist / T))   # forward speed, m/s

    heading = math.atan2(wp_y, max(wp_x, 1e-3))  # radians, + = left
    w       = max(-max_w, min(max_w, heading / T))  # angular velocity, rad/s

    speed_mm   = round(v * 1000)
    turn_degps = round(math.degrees(w))
    return f"drive:{speed_mm}:{turn_degps}"


# ── main inference loop ───────────────────────────────────────────────────────
async def inference_loop(hub: SensorHub, model: NoMaD, config: dict,
                         goal_tensor: torch.Tensor, goal_mask: torch.Tensor,
                         device: torch.device, chosen_wp: int,
                         executor: ThreadPoolExecutor, mode: str,
                         max_v: float, max_w: float, frame_rate: float):
    context_size = config["context_size"]
    frame_queue  = deque(maxlen=context_size + 1)
    loop         = asyncio.get_running_loop()
    step         = 0

    print("Waiting for iPhone connection...")
    while not hub.connected:
        await asyncio.sleep(0.2)
    print(f"iPhone connected — filling {context_size + 1}-frame buffer...")

    while True:
        jpeg = await hub.next_frame()
        img  = PILImage.open(BytesIO(jpeg)).convert("RGB")
        frame_queue.append(img)

        if len(frame_queue) < context_size + 1:
            print(f"  [{len(frame_queue)}/{context_size + 1}] buffering")
            continue

        waypoints, dist = await loop.run_in_executor(
            executor, _run_inference,
            list(frame_queue), goal_tensor, goal_mask, model, config, device,
            max_v, frame_rate
        )

        wp  = waypoints[chosen_wp]    # (x, y) metres
        cmd = _wp_to_drive(wp[0], wp[1], chosen_wp, max_v, max_w, frame_rate)
        step += 1

        if mode == "navigate":
            print(f"  [{step}] dist={dist:.1f}  wp[{chosen_wp}]=({wp[0]:.3f}m,{wp[1]:.3f}m) → {cmd}")
        else:
            print(f"  [{step}] wp[{chosen_wp}]=({wp[0]:.3f}m,{wp[1]:.3f}m) → {cmd}")

        try:
            await hub.send_hub_cmd(cmd)
        except RuntimeError:
            print("  Hub unreachable — skipping")


# ── startup ────────────────────────────────────────────────────────────────────
async def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    with open(args.robot_config) as f:
        robot = yaml.safe_load(f)
    max_v      = float(robot["max_v"])
    max_w      = float(robot["max_w"])
    frame_rate = float(robot["frame_rate"])
    print(f"Robot: max_v={max_v}m/s  max_w={max_w}rad/s  frame_rate={frame_rate}Hz")

    print(f"Loading NoMaD from {args.ckpt} ...")
    model = load_nomad(args.ckpt, config, device)
    print("Model ready.\n")

    image_size = config["image_size"][0]

    if args.mode == "navigate":
        goal_img    = PILImage.open(args.goal).convert("RGB")
        goal_tensor = _transform([goal_img], image_size).to(device)
        goal_mask   = torch.zeros(1, dtype=torch.long, device=device)
        print(f"Mode: NAVIGATE  goal={os.path.basename(args.goal)}")
    else:
        goal_tensor = torch.randn(1, 3, image_size, image_size, device=device)
        goal_mask   = torch.ones(1, dtype=torch.long, device=device)
        print("Mode: EXPLORE")

    hub      = SensorHub(port=args.port)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nomad")

    ip = local_ip()
    print(f"\nWebSocket server on port {args.port}")
    print(f"Connect iPhone app to:  ws://{ip}:{args.port}\n")

    server_task = asyncio.create_task(hub.serve_forever(), name="ws-server")
    infer_task  = asyncio.create_task(
        inference_loop(hub, model, config, goal_tensor, goal_mask,
                       device, args.waypoint, executor, args.mode,
                       max_v, max_w, frame_rate),
        name="inference"
    )

    try:
        await asyncio.gather(server_task, infer_task)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        server_task.cancel()
        infer_task.cancel()
        executor.shutdown(wait=False)
        if hub.connected:
            try:
                await hub.send_hub_cmd("drive:0:0")
            except Exception:
                pass
        print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode",     default="explore",
                        choices=["navigate", "explore"])
    parser.add_argument("--goal",     default=None,
                        help="Goal image path (navigate mode only)")
    parser.add_argument("--ckpt",         default=_DEFAULT_CKPT)
    parser.add_argument("--config",       default=_DEFAULT_CFG)
    parser.add_argument("--robot-config", default=_DEFAULT_ROBOT_CFG,
                        dest="robot_config")
    parser.add_argument("--port",         type=int, default=8765)
    parser.add_argument("--waypoint", type=int, default=2,
                        help="Waypoint index to act on (0-based, default 2)")
    args = parser.parse_args()

    if args.mode == "navigate" and args.goal is None:
        parser.error("--goal is required for navigate mode")

    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
