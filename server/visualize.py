#!/usr/bin/env python3
"""
Visualize ViNT waypoint predictions for a given obs/goal pair.

Produces a two-panel PNG:
  Left:  last observation frame with waypoints projected via a pinhole camera model
  Right: bird's-eye (top-down) view of the predicted waypoints

Camera projection parameters (--camera-h, --pitch-deg, --fov-h-deg) are
approximate. Tune them to match your actual camera/robot geometry.
"""
import argparse, os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch, yaml

_VISUALNAV = os.environ.get("VISUALNAV", os.path.expanduser("~/visualnav-transformer"))
sys.path.insert(0, os.path.join(_VISUALNAV, "train"))
from vint_train.models.vint.vint import ViNT

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def transform_images(pil_imgs, image_size):
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    tensors = []
    for img in pil_imgs:
        img = img.resize(image_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        tensors.append(t)
    return torch.cat(tensors, dim=1)


def load_images_from_dir(img_dir):
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
    imgs = [PILImage.open(os.path.join(img_dir, n)).convert("RGB") for n in names]
    return imgs, names


def load_vint(ckpt_path, config, device):
    model = ViNT(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        checkpoint = torch.load(ckpt_path, map_location=device)
    loaded = checkpoint["model"]
    try: state_dict = loaded.module.state_dict()
    except AttributeError: state_dict = loaded.state_dict()
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def project_waypoints(waypoints_xy, img_w, img_h, camera_h, pitch_deg, fov_h_deg):
    """
    Project (x, y) ground-plane waypoints (robot frame: x=forward, y=left)
    into image pixel coordinates using a pinhole camera model.
    Camera frame: X=right, Y=down, Z=forward.
    """
    cx, cy = img_w / 2, img_h / 2
    f = (img_w / 2) / np.tan(np.radians(fov_h_deg / 2))
    phi = np.radians(pitch_deg)

    pts = []
    for x, y in waypoints_xy:
        if x < 0.01:
            continue
        # Robot ground point → pre-rotation camera frame
        cam_X  = -y          # robot left  →  camera right (negated)
        cam_Y0 = camera_h    # ground is camera_h below camera  →  Y down = positive
        cam_Z0 = x           # forward distance

        # Apply downward pitch rotation (around camera X axis)
        cam_Z = cam_Z0 * np.cos(phi) + cam_Y0 * np.sin(phi)
        cam_Y = -cam_Z0 * np.sin(phi) + cam_Y0 * np.cos(phi)

        if cam_Z <= 0:
            continue

        u = f * cam_X / cam_Z + cx
        v = f * cam_Y / cam_Z + cy
        pts.append((u, v))
    return pts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--obs",    required=True, help="Directory of observation frames")
    parser.add_argument("--goal",   required=True, help="Goal image path")
    parser.add_argument("--out",    default="waypoints.png", help="Output PNG path")
    parser.add_argument("--ckpt",   default=os.path.join(os.path.dirname(__file__),
                                    "..", "model_weights", "vint.pth"))
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__),
                                    "..", "..", "train", "config", "vint.yaml"))
    parser.add_argument("--waypoint",   type=int,   default=2,
                        help="Index of the waypoint to act on (0-based)")
    _cam_cfg = os.path.join(os.path.dirname(__file__), "..", "config", "camera_iphone.yaml")
    parser.add_argument("--camera-config", default=_cam_cfg,
                        help="Camera YAML config (camera_h, pitch_deg, fov_h_deg)")
    # Per-run overrides — if set, take precedence over camera-config
    parser.add_argument("--camera-h",   type=float, default=None,
                        help="Override: camera height above ground (m)")
    parser.add_argument("--pitch-deg",  type=float, default=None,
                        help="Override: camera downward pitch (degrees)")
    parser.add_argument("--fov-h-deg",  type=float, default=None,
                        help="Override: camera horizontal FOV (degrees)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load camera params: file first, then CLI overrides
    with open(args.camera_config) as f:
        cam = yaml.safe_load(f)
    camera_h  = args.camera_h   if args.camera_h   is not None else cam["camera_h"]
    pitch_deg = args.pitch_deg  if args.pitch_deg  is not None else cam["pitch_deg"]
    fov_h_deg = args.fov_h_deg  if args.fov_h_deg  is not None else cam["fov_h_deg"]
    image_size   = config["image_size"]    # [width, height] for model
    context_size = config["context_size"]
    max_v        = 0.2   # m/s
    frame_rate   = 4     # Hz

    model = load_vint(args.ckpt, config, device)

    all_imgs, _ = load_images_from_dir(args.obs)
    obs_imgs  = all_imgs[-(context_size + 1):]
    goal_img  = PILImage.open(args.goal).convert("RGB")

    obs_tensor  = transform_images(obs_imgs,  image_size).to(device)
    goal_tensor = transform_images([goal_img], image_size).to(device)

    with torch.no_grad():
        dist_pred, action_pred = model(obs_tensor, goal_tensor)

    dist      = dist_pred.item()
    waypoints = action_pred[0].cpu().numpy()   # [len_traj_pred, 4]
    if config["normalize"]:
        waypoints[:, :2] *= (max_v / frame_rate)

    # Waypoints are cumulative positions in robot egocentric frame
    wp_xy = waypoints[:, :2]   # [[x0,y0], [x1,y1], ...]

    # ------------------------------------------------------------------ #
    # Figure                                                               #
    # ------------------------------------------------------------------ #
    fig, (ax_cam, ax_top) = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("#1a1a1a")
    for ax in (ax_cam, ax_top):
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    goal_label = os.path.basename(args.goal)
    fig.suptitle(
        f"ViNT  ·  distance to goal ({goal_label}): {dist:.2f} steps",
        fontsize=13, fontweight="bold", color="white"
    )

    # ------------------------------------------------------------------ #
    # Left: camera view                                                    #
    # ------------------------------------------------------------------ #
    last_obs = obs_imgs[-1]
    img_w, img_h = last_obs.size
    ax_cam.imshow(np.array(last_obs))
    ax_cam.set_title("Last obs frame  +  projected waypoints\n"
                      f"(camera h={camera_h}m  pitch={pitch_deg}°  FOV={fov_h_deg}°  — approximate)",
                      color="white", fontsize=9)
    ax_cam.axis("off")

    # Project robot origin (very small x to get the foot point)
    origin_proj = project_waypoints([(0.005, 0.0)], img_w, img_h,
                                    camera_h, pitch_deg, fov_h_deg)
    foot = origin_proj[0] if origin_proj else (img_w / 2, img_h * 0.92)

    proj_pts = project_waypoints(wp_xy, img_w, img_h,
                                 camera_h, pitch_deg, fov_h_deg)

    if proj_pts:
        all_cam = [foot] + proj_pts
        ux = [p[0] for p in all_cam]
        uy = [p[1] for p in all_cam]

        # All waypoints: yellow
        ax_cam.plot(ux, uy, "o-", color="yellow", linewidth=2.5,
                    markersize=5, zorder=3, label="waypoints")

        # Chosen waypoint: blue line from foot, green dot
        chosen_idx = min(args.waypoint, len(proj_pts) - 1)
        cx_pt, cy_pt = proj_pts[chosen_idx]
        ax_cam.plot([foot[0], cx_pt], [foot[1], cy_pt], "-",
                    color="royalblue", linewidth=3.5, zorder=4)
        ax_cam.plot(cx_pt, cy_pt, "o", color="limegreen", markersize=10, zorder=5)

    # ------------------------------------------------------------------ #
    # Right: bird's-eye view                                               #
    # ------------------------------------------------------------------ #
    ax_top.set_title("Bird's-eye view  (robot at origin, facing up)",
                     color="white", fontsize=9)

    # Grid
    max_x = max(wp_xy[:, 0].max() * 1.3, 0.5)
    max_y = max(abs(wp_xy[:, 1]).max() * 1.5, 0.3)
    for xi in np.arange(-max_y * 1.2, max_y * 1.2 + 0.01, 0.1):
        ax_top.axvline(xi, color="#2a2a2a", linewidth=0.6)
    for yi in np.arange(0, max_x * 1.1, 0.1):
        ax_top.axhline(yi, color="#2a2a2a", linewidth=0.6)

    # Robot marker
    robot = plt.Polygon([(-0.015, -0.02), (0.015, -0.02), (0, 0.03)],
                        closed=True, color="gray", zorder=5)
    ax_top.add_patch(robot)

    # Trajectory: negate y so robot-right appears on the right of the plot.
    # Robot frame: y positive = LEFT. Plot convention: positive x = RIGHT.
    xs_plot = np.insert(-wp_xy[:, 1], 0, 0.0)   # -y → horiz (right = robot right)
    ys_plot = np.insert( wp_xy[:, 0], 0, 0.0)   #  x → vert  (up   = forward)

    ax_top.plot(xs_plot, ys_plot, "o-", color="gold", linewidth=2.5,
                markersize=7, zorder=3, label="waypoints")

    chosen_idx = min(args.waypoint, len(wp_xy) - 1)
    ax_top.plot([0, -wp_xy[chosen_idx, 1]], [0, wp_xy[chosen_idx, 0]],
                "-", color="royalblue", linewidth=3, zorder=4)
    ax_top.plot(-wp_xy[chosen_idx, 1], wp_xy[chosen_idx, 0],
                "o", color="limegreen", markersize=10, zorder=6,
                label=f"chosen [{chosen_idx}]")

    ax_top.set_xlim(-max_y * 1.3, max_y * 1.3)
    ax_top.set_ylim(-0.05, max_x * 1.15)
    ax_top.set_aspect("equal")
    ax_top.set_xlabel("lateral (m)  ← left  |  right →", color="white")
    ax_top.set_ylabel("forward (m)", color="white")
    ax_top.legend(loc="upper right", fontsize=8,
                  facecolor="#2a2a2a", labelcolor="white", edgecolor="#555")

    # Inset: goal thumbnail
    goal_ax = fig.add_axes([0.91, 0.65, 0.075, 0.28])
    goal_ax.imshow(np.array(goal_img))
    goal_ax.set_title("goal", fontsize=8, color="white")
    goal_ax.axis("off")

    plt.tight_layout(rect=[0, 0, 0.91, 0.95])
    plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
