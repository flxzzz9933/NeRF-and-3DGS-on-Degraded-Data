#!/usr/bin/env python3
"""
Prepare degraded NeRF Synthetic (Blender) dataset.

- 输入:  Dataset/nerf_synthetic/<scene>/{train,val,test} + transforms_*.json
- 输出:  <output_root>/<scene>/...
  * train: 视角子采样 + blur + 曝光扰动 + 可选 pose noise
  * val/test: 原样复制
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def list_scenes(root: Path, scenes: List[str] | None):
    if scenes:
        return [root / s for s in scenes]
    return sorted([p for p in root.iterdir() if p.is_dir()])


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def choose_subset(frames, fraction: float, seed: int):
    if fraction >= 0.999:
        return frames
    n = len(frames)
    k = max(1, int(round(n * fraction)))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=k, replace=False))
    return [frames[i] for i in idx.tolist()]


def perturb_pose_matrix(matrix, rot_deg_std: float, trans_std: float,
                        rng: np.random.Generator):
    if rot_deg_std <= 0.0 and trans_std <= 0.0:
        return matrix

    T = np.array(matrix, dtype=np.float32)
    R = T[:3, :3]
    t = T[:3, 3]

    # rotation noise
    if rot_deg_std > 0.0:
        angle = np.deg2rad(rng.normal(0.0, rot_deg_std))
        if abs(angle) > 1e-6:
            axis = rng.normal(size=3)
            axis /= (np.linalg.norm(axis) + 1e-8)
            x, y, z = axis
            K = np.array([[0, -z, y],
                          [z, 0, -x],
                          [-y, x, 0]], dtype=np.float32)
            I = np.eye(3, dtype=np.float32)
            R_delta = I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            R = R_delta @ R

    # translation noise
    if trans_std > 0.0:
        t = t + rng.normal(scale=trans_std, size=3)

    T[:3, :3] = R
    T[:3, 3] = t
    return T.tolist()


def degrade_image(img: Image.Image, rng: np.random.Generator,
                  blur_radius: float, exposure_std: float):
    if blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    if exposure_std > 0.0:
        factor = float(rng.normal(1.0, exposure_std))
        factor = max(0.3, min(1.8, factor))
        img = ImageEnhance.Brightness(img).enhance(factor)
    return img


def process_scene(scene_dir: Path, out_root: Path,
                  view_fraction: float,
                  blur_radius: float,
                  exposure_std: float,
                  rot_deg_std: float,
                  trans_std: float,
                  seed: int):
    scene = scene_dir.name
    print(f"\n[SCENE] {scene}")

    tf_train_path = scene_dir / "transforms_train.json"
    if not tf_train_path.exists():
        print("  No transforms_train.json, skip.")
        return
    tf_val_path = scene_dir / "transforms_val.json"
    tf_test_path = scene_dir / "transforms_test.json"

    tf_train = load_json(tf_train_path)
    frames = tf_train.get("frames", [])
    print(f"  Train frames: {len(frames)}")

    frames_sub = choose_subset(frames, view_fraction, seed)
    print(f"  After subsampling: {len(frames_sub)} frames")

    rng = np.random.default_rng(seed)

    out_scene = out_root / scene
    train_out = out_scene / "train"
    val_out = out_scene / "val"
    test_out = out_scene / "test"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)
    test_out.mkdir(parents=True, exist_ok=True)

    def resolve_image_path(file_path: str) -> Path:
        # file_path 形如 "train/r_0"
        for ext in [".png", ".jpg", ".jpeg"]:
            p = scene_dir / (file_path + ext)
            if p.exists():
                return p
        raise FileNotFoundError(f"Cannot find image for {file_path} in {scene_dir}")

    # 处理 train（降质 + pose noise）
    new_frames = []
    for fr in frames_sub:
        fp = fr["file_path"]    # e.g. "train/r_0"
        src_img = resolve_image_path(fp)
        img = Image.open(src_img).convert("RGB")
        img = degrade_image(img, rng, blur_radius, exposure_std)
        dst_img = train_out / src_img.name
        img.save(dst_img)

        fr = dict(fr)
        if "transform_matrix" in fr:
            fr["transform_matrix"] = perturb_pose_matrix(
                fr["transform_matrix"], rot_deg_std, trans_std, rng
            )
        # file_path 保持和原来规则一致，只是目录固定成 train
        fr["file_path"] = "train/" + src_img.stem
        new_frames.append(fr)

    tf_train_new = dict(tf_train)
    tf_train_new["frames"] = new_frames
    save_json(tf_train_new, out_scene / "transforms_train.json")

    # val/test 原样复制
    if tf_val_path.exists():
        save_json(load_json(tf_val_path), out_scene / "transforms_val.json")
        for p in (scene_dir / "val").iterdir():
            if p.suffix in IMAGE_EXTS:
                shutil.copy2(p, val_out / p.name)

    if tf_test_path.exists():
        save_json(load_json(tf_test_path), out_scene / "transforms_test.json")
        for p in (scene_dir / "test").iterdir():
            if p.suffix in IMAGE_EXTS:
                shutil.copy2(p, test_out / p.name)

    print(f"  Saved degraded scene to {out_scene}")


def main():
    ap = argparse.ArgumentParser(
        description="Degrade NeRF Synthetic (Blender) dataset."
    )
    ap.add_argument("--input-root", type=str, required=True,
                    help="Root of nerf_synthetic dataset.")
    ap.add_argument("--output-root", type=str, required=True,
                    help="Output root for degraded dataset.")
    ap.add_argument("--view-fraction", type=float, default=1.0,
                    help="Fraction of TRAIN views to keep.")
    ap.add_argument("--blur-radius", type=float, default=0.0,
                    help="Gaussian blur radius.")
    ap.add_argument("--exposure-std", type=float, default=0.0,
                    help="Std for brightness factor N(1, std).")
    ap.add_argument("--rot-deg-std", type=float, default=0.0,
                    help="Std of pose rotation noise (degrees).")
    ap.add_argument("--trans-std", type=float, default=0.0,
                    help="Std of pose translation noise (scene units).")
    ap.add_argument("--scenes", type=str, nargs="*", default=None,
                    help="Optional scene names (default: all).")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    in_root = Path(args.input_root).expanduser().resolve()
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    scenes = list_scenes(in_root, args.scenes)
    print("Found scenes:", [s.name for s in scenes])
    print("Output root:", out_root)

    for s in scenes:
        process_scene(
            s,
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
