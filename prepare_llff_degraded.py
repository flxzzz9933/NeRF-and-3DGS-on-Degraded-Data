#!/usr/bin/env python3
"""
Prepare degraded LLFF datasets for NeRF / 3DGS robustness experiments.

- Loads all LLFF scenes under --input-root.
- For each scene:
    * loads poses_bounds.npy and images
    * optionally subsamples views (view_fraction)
    * applies image blur + exposure jitter
    * applies small pose rotation + translation noise
- Saves a LLFF-compatible scene to --output-root.

This is the "Stage 1: dataset prep + degradations" part of your project.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".JPG", ".PNG", ".JPEG"}


def find_image_dir(scene_dir: Path) -> Path:
    """Try common LLFF image folders in priority order."""
    for name in ["images_4", "images_2", "images"]:
        d = scene_dir / name
        if d.exists() and d.is_dir():
            return d
    raise FileNotFoundError(
        f"No images[_2|_4] folder found in {scene_dir}"
    )


def list_scenes(root: Path, explicit: List[str] = None) -> List[Path]:
    """List scene directories under root (optionally filter by name)."""
    if explicit:
        return [root / s for s in explicit]
    return sorted([p for p in root.iterdir() if p.is_dir()])


def load_llff_scene(scene_dir: Path) -> Tuple[List[Path], np.ndarray]:
    """Load image file paths and poses_bounds.npy for a LLFF scene."""
    pose_file = scene_dir / "poses_bounds.npy"
    if not pose_file.exists():
        raise FileNotFoundError(f"{pose_file} not found")

    poses_bounds = np.load(pose_file)  # (N, 17)
    img_dir = find_image_dir(scene_dir)
    img_paths = sorted(
        p for p in img_dir.iterdir() if p.suffix in IMAGE_EXTS
    )

    if len(img_paths) != poses_bounds.shape[0]:
        print(
            f"[WARN] {scene_dir.name}: {len(img_paths)} images vs "
            f"{poses_bounds.shape[0]} poses"
        )

    return img_paths, poses_bounds


def subsample_views(
    img_paths: List[Path],
    poses_bounds: np.ndarray,
    fraction: float,
    seed: int = 0,
) -> Tuple[List[Path], np.ndarray]:
    """Randomly subsample a fraction of views."""
    if fraction >= 0.999:
        return img_paths, poses_bounds

    n = len(img_paths)
    k = max(1, int(round(n * fraction)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    idx = np.sort(idx)

    img_paths_sub = [img_paths[i] for i in idx]
    poses_bounds_sub = poses_bounds[idx]

    return img_paths_sub, poses_bounds_sub


def apply_image_degradations(
    img: Image.Image,
    rng: np.random.Generator,
    blur_radius: float = 0.0,
    exposure_std: float = 0.0,
) -> Image.Image:
    """Apply Gaussian blur and exposure jitter to a PIL image."""
    if blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    if exposure_std > 0.0:
        # Brightness factor ~ N(1.0, exposure_std), clamped.
        factor = float(rng.normal(loc=1.0, scale=exposure_std))
        factor = max(0.3, min(1.8, factor))
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)

    return img


def add_pose_noise_llff(
    poses_bounds: np.ndarray,
    rot_deg_std: float = 0.0,
    trans_std: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Add small rotation / translation noise to LLFF poses.

    LLFF stores first 15 numbers as 3x5 matrix: [R(3x3) | t(3) | hwf].
    We only perturb R and t, keep bounds / hwf untouched.
    """
    if rot_deg_std <= 0.0 and trans_std <= 0.0:
        return poses_bounds

    rng = np.random.default_rng(seed)
    pb = poses_bounds.copy()
    poses = pb[:, :15].reshape(-1, 3, 5)
    bounds = pb[:, 15:]

    for i in range(poses.shape[0]):
        R = poses[i, :, :3]
        t = poses[i, :, 3]

        # Rotation noise (single small axis-angle)
        if rot_deg_std > 0.0:
            angle = np.deg2rad(rng.normal(0.0, rot_deg_std))
            axis = rng.normal(size=3)
            axis_norm = np.linalg.norm(axis) + 1e-8
            axis = axis / axis_norm

            x, y, z = axis
            K = np.array(
                [
                    [0, -z, y],
                    [z, 0, -x],
                    [-y, x, 0],
                ],
                dtype=np.float32,
            )
            I = np.eye(3, dtype=np.float32)
            R_delta = (
                I
                + np.sin(angle) * K
                + (1.0 - np.cos(angle)) * (K @ K)
            )
            R = R_delta @ R

        # Translation noise
        if trans_std > 0.0:
            t = t + rng.normal(scale=trans_std, size=3)

        poses[i, :, :3] = R
        poses[i, :, 3] = t

    pb_noisy = np.concatenate(
        [poses.reshape(-1, 15), bounds], axis=1
    )
    return pb_noisy


def save_degraded_scene(
    scene_name: str,
    img_paths: List[Path],
    poses_bounds: np.ndarray,
    out_root: Path,
    blur_radius: float,
    exposure_std: float,
    seed: int,
):
    """
    Save degraded images + poses to out_root/scene_name.

    Structure:
        out_root/scene_name/poses_bounds.npy
        out_root/scene_name/images/
    """
    out_scene = out_root / scene_name
    out_img_dir = out_scene / "images"
    out_scene.mkdir(parents=True, exist_ok=True)
    out_img_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    for src_path in img_paths:
        img = Image.open(src_path).convert("RGB")
        img = apply_image_degradations(
            img, rng, blur_radius=blur_radius, exposure_std=exposure_std
        )

        dst_path = out_img_dir / src_path.name
        img.save(dst_path)

    np.save(out_scene / "poses_bounds.npy", poses_bounds)


def process_scene(
    scene_dir: Path,
    out_root: Path,
    view_fraction: float,
    blur_radius: float,
    exposure_std: float,
    rot_deg_std: float,
    trans_std: float,
    seed: int,
):
    print(f"\n[SCENE] {scene_dir.name}")
    img_paths, poses_bounds = load_llff_scene(scene_dir)
    print(f"  Loaded {len(img_paths)} images")

    img_paths, poses_bounds = subsample_views(
        img_paths, poses_bounds, view_fraction, seed=seed
    )
    print(f"  After subsampling: {len(img_paths)} images")

    # Pose noise
    poses_bounds_noisy = add_pose_noise_llff(
        poses_bounds,
        rot_deg_std=rot_deg_std,
        trans_std=trans_std,
        seed=seed,
    )

    save_degraded_scene(
        scene_dir.name,
        img_paths,
        poses_bounds_noisy,
        out_root,
        blur_radius=blur_radius,
        exposure_std=exposure_std,
        seed=seed,
    )
    print(f"  Saved degraded scene to {out_root/scene_dir.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare degraded LLFF datasets."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root directory containing LLFF scenes "
        "(e.g., Dataset/llff).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Output root for degraded scenes.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of scene names. "
        "If omitted, all subdirectories are used.",
    )
    parser.add_argument(
        "--view-fraction",
        type=float,
        default=1.0,
        help="Fraction of views to keep (1.0, 0.5, 0.25, 0.1, ...).",
    )
    parser.add_argument(
        "--blur-radius",
        type=float,
        default=0.0,
        help="Gaussian blur radius in pixels (0 = no blur).",
    )
    parser.add_argument(
        "--exposure-std",
        type=float,
        default=0.0,
        help="Std dev for brightness factor N(1, std).",
    )
    parser.add_argument(
        "--rot-deg-std",
        type=float,
        default=0.0,
        help="Std dev of rotation noise in degrees.",
    )
    parser.add_argument(
        "--trans-std",
        type=float,
        default=0.0,
        help="Std dev of translation noise (same units as poses).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling & noise.",
    )

    args = parser.parse_args()

    in_root = Path(args.input_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = list_scenes(in_root, args.scenes)
    if not scenes:
        raise RuntimeError(f"No scenes found under {in_root}")

    print(f"Found scenes: {[s.name for s in scenes]}")
    print(f"Output root: {out_root}")

    for scene_dir in scenes:
        process_scene(
            scene_dir,
            out_root,
            view_fraction=args.view_fraction,
            blur_radius=args.blur_radius,
            exposure_std=args.exposure_std,
            rot_deg_std=args.rot_deg_std,
            trans_std=args.trans_std,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
