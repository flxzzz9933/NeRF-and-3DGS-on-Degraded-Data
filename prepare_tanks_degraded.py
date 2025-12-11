#!/usr/bin/env python3
"""
Prepare degraded Tanks & Temples datasets (Truck etc.) for NeRF / 3DGS experiments.

This version degrades **all three axes C / I / P**:

- Coverage C:    random view subsampling (view_fraction)
- Image Quality I: Gaussian blur + exposure (brightness) jitter
- Pose Noise P:   small SE(3) noise on the 4x4 transform_matrix in transforms.json

Assumptions / pipeline:

1. You have already run COLMAP once via `ns-process-data images` to get a
   *baseline* Nerfstudio dataset:

   ns_ready/tanks/<Scene>/baseline/
       images/...
       transforms.json

2. This script **does NOT** rerun COLMAP. It:
   - loads the baseline transforms.json
   - subsamples frames
   - adds pose noise
   - copies & degrades the corresponding images
   - writes a new Nerfstudio-style dataset to:

   <output_root>/<Scene>/
       images/...
       transforms.json

So you can point `ns-train` or 3DGS code directly at
`nerf_data/tanks_degraded/T0/Truck`, etc.
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def list_scenes(root: Path, scenes: List[str] | None) -> list[str]:
    """Return list of scene names to process."""
    if scenes:
        return scenes
    # fallback: list directories under baseline root
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def load_transforms(baseline_root: Path, scene: str) -> dict:
    """Load baseline transforms.json for a scene."""
    tf_path = baseline_root / scene / "baseline" / "transforms.json"
    if not tf_path.exists():
        raise FileNotFoundError(f"Baseline transforms not found: {tf_path}")
    with open(tf_path, "r") as f:
        data = json.load(f)
    if "frames" not in data or not isinstance(data["frames"], list):
        raise ValueError(f"Malformed transforms.json at {tf_path}")
    return data


def subsample_frames(frames: list[dict], fraction: float, seed: int) -> list[dict]:
    """Randomly subsample a fraction of frames."""
    if fraction >= 0.999:
        return frames
    n = len(frames)
    if n == 0:
        return frames
    k = max(1, int(round(n * fraction)))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    return [frames[i] for i in idx]


def random_rotation_matrix(rng: np.random.Generator, rot_deg_std: float) -> np.ndarray:
    """Small random rotation ~ N(0, rot_deg_std) per axis, in degrees."""
    if rot_deg_std <= 0.0:
        return np.eye(3, dtype=np.float32)
    sigma = np.deg2rad(rot_deg_std)
    rx, ry, rz = rng.normal(0.0, sigma, size=3)

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0, 1]], dtype=np.float32)

    # Compose as Rz * Ry * Rx (extrinsic X-Y-Z)
    return (Rz @ Ry @ Rx).astype(np.float32)


def add_pose_noise(
    frames: list[dict],
    rot_deg_std: float,
    trans_std: float,
    seed: int,
) -> list[dict]:
    """Add SE(3) noise to transform_matrix in frames."""
    if rot_deg_std <= 0.0 and trans_std <= 0.0:
        return frames

    rng = np.random.default_rng(seed)
    noisy_frames: list[dict] = []

    for f in frames:
        if "transform_matrix" not in f:
            noisy_frames.append(f)
            continue

        T = np.array(f["transform_matrix"], dtype=np.float32)  # (4,4)
        if T.shape != (4, 4):
            noisy_frames.append(f)
            continue

        R = T[:3, :3]
        t = T[:3, 3]

        # Rotation noise
        R_noise = random_rotation_matrix(rng, rot_deg_std)
        R_new = R_noise @ R

        # Translation noise
        if trans_std > 0.0:
            t_noise = rng.normal(0.0, trans_std, size=3).astype(np.float32)
            t_new = t + t_noise
        else:
            t_new = t

        T_new = np.eye(4, dtype=np.float32)
        T_new[:3, :3] = R_new
        T_new[:3, 3] = t_new

        f_new = dict(f)
        f_new["transform_matrix"] = T_new.tolist()
        noisy_frames.append(f_new)

    return noisy_frames


def degrade_image(
    img: Image.Image,
    rng: np.random.Generator,
    blur_radius: float,
    exposure_std: float,
) -> Image.Image:
    """Apply blur + brightness jitter."""
    if blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if exposure_std > 0.0:
        factor = float(rng.normal(1.0, exposure_std))
        factor = max(0.3, min(1.8, factor))
        img = ImageEnhance.Brightness(img).enhance(factor)

    return img


def save_degraded_scene(
    baseline_root: Path,
    out_root: Path,
    scene: str,
    frames: list[dict],
    blur_radius: float,
    exposure_std: float,
    seed: int,
    meta: dict,
):
    """Copy + degrade images and write new transforms.json."""
    out_scene = out_root / scene
    out_scene.mkdir(parents=True, exist_ok=True)
    out_img_root = out_scene  # we'll respect file_path subdirs (e.g. images/...)

    rng = np.random.default_rng(seed)

    for f in frames:
        fp = f.get("file_path")
        if fp is None:
            continue
        rel = Path(fp)
        src = baseline_root / scene / "baseline" / rel
        if not src.exists():
            raise FileNotFoundError(f"Image not found: {src}")

        dst = out_img_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        img = Image.open(src).convert("RGB")
        img = degrade_image(img, rng, blur_radius, exposure_std)
        img.save(dst)

    # write transforms.json
    out_tf = dict(meta)
    out_tf["frames"] = frames
    tf_path = out_scene / "transforms.json"
    with open(tf_path, "w") as f:
        json.dump(out_tf, f, indent=2)
    print(f"  Wrote transforms.json -> {tf_path}")


def process_scene(
    scene: str,
    baseline_root: Path,
    out_root: Path,
    view_fraction: float,
    blur_radius: float,
    exposure_std: float,
    rot_deg_std: float,
    trans_std: float,
    seed: int,
):
    print(f"\n[SCENE] {scene}")
    transforms = load_transforms(baseline_root, scene)
    frames = transforms["frames"]
    print(f"  Loaded {len(frames)} frames from baseline transforms.json")

    frames_sub = subsample_frames(frames, view_fraction, seed=seed)
    print(f"  After subsampling: {len(frames_sub)} frames")

    frames_noisy = add_pose_noise(
        frames_sub,
        rot_deg_std=rot_deg_std,
        trans_std=trans_std,
        seed=seed,
    )
    print(
        f"  Pose noise: rot_std={rot_deg_std} deg, "
        f"trans_std={trans_std}"
    )

    save_degraded_scene(
        baseline_root=baseline_root,
        out_root=out_root,
        scene=scene,
        frames=frames_noisy,
        blur_radius=blur_radius,
        exposure_std=exposure_std,
        seed=seed,
        meta=transforms,
    )
    print(f"  Saved degraded scene to {out_root/scene}")


def main():
    ap = argparse.ArgumentParser(
        description="Degrade Tanks&Temples baseline Nerfstudio datasets (C/I/P)."
    )
    ap.add_argument(
        "--baseline-root",
        type=str,
        default="ns_ready/tanks",
        help="Root containing <Scene>/baseline/transforms.json "
             "(output of ns-process-data images).",
    )
    ap.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root for degraded sets, e.g. nerf_data/tanks_degraded/T0",
    )
    ap.add_argument(
        "--view-fraction",
        type=float,
        default=1.0,
        help="Fraction of views to keep (0–1).",
    )
    ap.add_argument(
        "--blur-radius",
        type=float,
        default=0.0,
        help="Gaussian blur radius in pixels.",
    )
    ap.add_argument(
        "--exposure-std",
        type=float,
        default=0.0,
        help="Std dev for brightness factor N(1, std).",
    )
    ap.add_argument(
        "--rot-deg-std",
        type=float,
        default=0.0,
        help="Std dev of rotation noise in degrees.",
    )
    ap.add_argument(
        "--trans-std",
        type=float,
        default=0.0,
        help="Std dev of translation noise in world units.",
    )
    ap.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of scene names (e.g. Truck). "
             "If omitted, all scenes under baseline-root are used.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling & jitter.",
    )

    args = ap.parse_args()

    baseline_root = Path(args.baseline_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = list_scenes(baseline_root, args.scenes)
    print("Found scenes:", scenes)
    print("Baseline root:", baseline_root)
    print("Output root:", out_root)

    for scene in scenes:
        process_scene(
            scene=scene,
            baseline_root=baseline_root,
            out_root=out_root,
            view_fraction=args.view_fraction,
            blur_radius=args.blur_radius,
            exposure_std=args.exposure_std,
            rot_deg_std=args.rot_deg_std,
            trans_std=args.trans_std,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
