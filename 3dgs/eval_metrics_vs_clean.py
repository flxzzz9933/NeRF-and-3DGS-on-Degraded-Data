#!/usr/bin/env python
import os
import sys
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print("[FATAL] scikit-image not installed. Run: pip install scikit-image")
    sys.exit(1)

try:
    import torch
    import lpips
except ImportError:
    print("[FATAL] lpips or torch not installed. Run: pip install lpips")
    sys.exit(1)


def psnr(img_pred, img_gt):
    img_pred = np.float32(img_pred)
    img_gt = np.float32(img_gt)
    mse = np.mean((img_pred - img_gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def make_lpips_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = lpips.LPIPS(net="vgg").to(device)
    net.eval()
    return net, device


LPIPS_NET, DEVICE = make_lpips_model()


def img_to_lpips_tensor(img_np):
    """HWC uint8 -> 1x3xHxW float32 in [-1,1] on DEVICE"""
    t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t * 2.0 - 1.0
    return t.to(DEVICE)


def eval_from_pairs(pairs):
    """pairs: list of (render_path, clean_path)"""
    all_psnr = []
    all_ssim = []
    all_lpips = []

    for r_path, c_path in pairs:
        img_r = np.array(Image.open(r_path).convert("RGB"))
        img_c = np.array(Image.open(c_path).convert("RGB"))

        if img_r.shape != img_c.shape:
            # 保险起见：把 render resize 到 clean 的分辨率
            img_r = np.array(
                Image.fromarray(img_r).resize(
                    (img_c.shape[1], img_c.shape[0]),
                    Image.BILINEAR,
                )
            )

        p = psnr(img_r, img_c)
        s = ssim(img_c, img_r, data_range=255, channel_axis=-1)

        t_r = img_to_lpips_tensor(img_r)
        t_c = img_to_lpips_tensor(img_c)
        with torch.no_grad():
            l = LPIPS_NET(t_r, t_c).item()

        all_psnr.append(p)
        all_ssim.append(s)
        all_lpips.append(l)

    if not all_psnr:
        print("No valid pairs, nothing to evaluate.")
        return

    print(f"Num images used: {len(all_psnr)}")
    print(f"Mean PSNR:  {np.mean(all_psnr):.3f} dB")
    print(f"Mean SSIM:  {np.mean(all_ssim):.4f}")
    print(f"Mean LPIPS: {np.mean(all_lpips):.4f}")


def build_pairs_truck(renders_dir, clean_root, sparse_dir):
    """
    Truck 模式：
      renders_dir: 3DGS 渲染目录 (…/renders)
      clean_root:  clean 图像目录 (…/tanks/image_sets/Truck)
      sparse_dir:  T0/T1/T2 的 sparse/0
    """
    images_txt = Path(sparse_dir) / "images.txt"
    if not images_txt.is_file():
        print(f"[FATAL] images.txt not found in {sparse_dir}")
        sys.exit(1)

    # 读取 COLMAP images.txt，按 IMAGE_ID 排序
    colmap_names = []
    with images_txt.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # 第一行: IMAGE_ID qw qx qy qz tx ty tz camera_id name
            if len(parts) >= 10:
                name = parts[9]
                colmap_names.append(name)
                # 下一行是 POINTS2D，直接忽略

    render_files = sorted(
        f for f in os.listdir(renders_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not render_files:
        print(f"[FATAL] No render images in {renders_dir}")
        sys.exit(1)

    n = min(len(render_files), len(colmap_names))
    print(f"[INFO] renders={len(render_files)}, colmap_images={len(colmap_names)}, using n={n}")

    pairs = []
    for i in range(n):
        render_name = render_files[i]
        colmap_name = os.path.basename(colmap_names[i])
        clean_path = os.path.join(clean_root, colmap_name)
        if not os.path.isfile(clean_path):
            print(f"[WARN] Clean image not found for {colmap_name}, skip")
            continue

        r_path = os.path.join(renders_dir, render_name)
        pairs.append((r_path, clean_path))

    return pairs


def build_pairs_nerf(renders_dir, clean_root, degraded_root, split):
    """
    NeRF-synthetic 模式：
      renders_dir: 3DGS 渲染目录 (…/train/ours_10000/renders)
      clean_root:  clean nerf_synthetic scene 根目录 (…/nerf_synthetic/chair)
      degraded_root:  对应的 degrade 子目录 (…/nerf_synthetic_degraded/chair/E0.../chair)
      split: train / test / val
    """
    tf_path = Path(degraded_root) / f"transforms_{split}.json"
    if not tf_path.is_file():
        print(f"[FATAL] transforms file not found: {tf_path}")
        sys.exit(1)

    with tf_path.open("r") as f:
        tf = json.load(f)

    frames = tf.get("frames", [])
    if not frames:
        print(f"[FATAL] No frames in {tf_path}")
        sys.exit(1)

    render_files = sorted(
        f for f in os.listdir(renders_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not render_files:
        print(f"[FATAL] No render images in {renders_dir}")
        sys.exit(1)

    n = min(len(frames), len(render_files))
    print(f"[INFO] frames={len(frames)}, renders={len(render_files)}, using n={n}")

    pairs = []
    for i in range(n):
        frame = frames[i]
        fp = frame.get("file_path", "")
        if not fp:
            print(f"[WARN] frame {i} has no file_path, skip")
            continue

        clean_path = os.path.join(clean_root, fp + ".png")
        if not os.path.isfile(clean_path):
            print(f"[WARN] Clean image not found for {fp}.png, skip")
            continue

        r_path = os.path.join(renders_dir, render_files[i])
        pairs.append((r_path, clean_path))

    return pairs


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Truck (COLMAP) 模式:")
        print("    python eval_metrics_vs_clean.py truck "
              "<renders_dir> <clean_dir> <sparse_dir>")
        print("")
        print("  NeRF-synthetic 模式:")
        print("    python eval_metrics_vs_clean.py nerf "
              "<renders_dir> <clean_root> <degraded_root> [split]")
        print("    split 默认 train，可选 train/test/val")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "truck":
        if len(sys.argv) != 5:
            print("Usage: python eval_metrics_vs_clean.py truck "
                  "<renders_dir> <clean_dir> <sparse_dir>")
            sys.exit(1)
        renders_dir = os.path.expanduser(sys.argv[2])
        clean_dir = os.path.expanduser(sys.argv[3])
        sparse_dir = os.path.expanduser(sys.argv[4])
        pairs = build_pairs_truck(renders_dir, clean_dir, sparse_dir)
        eval_from_pairs(pairs)

    elif mode == "nerf":
        if len(sys.argv) < 5:
            print("Usage: python eval_metrics_vs_clean.py nerf "
                  "<renders_dir> <clean_root> <degraded_root> [split]")
            sys.exit(1)
        renders_dir = os.path.expanduser(sys.argv[2])
        clean_root = os.path.expanduser(sys.argv[3])
        degraded_root = os.path.expanduser(sys.argv[4])
        split = sys.argv[5] if len(sys.argv) > 5 else "train"
        pairs = build_pairs_nerf(renders_dir, clean_root, degraded_root, split)
        eval_from_pairs(pairs)

    else:
        print(f"[FATAL] Unknown mode: {mode} (expected 'truck' or 'nerf')")
        sys.exit(1)


if __name__ == "__main__":
    main()
