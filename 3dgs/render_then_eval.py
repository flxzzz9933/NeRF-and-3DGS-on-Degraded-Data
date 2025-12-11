import os
import sys
import math
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def read_image_names_from_colmap(images_txt_path):
    names = []
    with open(images_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            last = parts[-1].lower()
            # 只要最后一个 token 是 jpg/png 才当作一张图
            if last.endswith((".jpg", ".jpeg", ".png")):
                names.append(os.path.basename(parts[-1]))
    return names


def find_clean_image(clean_dir, base_name):
    """
    尝试在 clean_dir 里找到对应的干净图像：
    1. 原始文件名
    2. 替换扩展名 .jpg/.jpeg/.png
    """
    clean_dir = Path(clean_dir)
    base_name = os.path.basename(base_name)
    stem, ext = os.path.splitext(base_name)

    candidates = [
        clean_dir / base_name,
        clean_dir / f"{stem}.jpg",
        clean_dir / f"{stem}.jpeg",
        clean_dir / f"{stem}.png",
    ]

    for p in candidates:
        if p.exists():
            return p
    return None


def eval_vs_clean_colmap(renders_dir, clean_images_dir, sparse_dir):
    renders_dir = Path(renders_dir)
    clean_images_dir = Path(clean_images_dir)
    sparse_dir = Path(sparse_dir)

    render_files = sorted(renders_dir.glob("*.png"))
    if not render_files:
        raise RuntimeError(f"No PNG renders found in {renders_dir}")

    images_txt = sparse_dir / "images.txt"
    if not images_txt.exists():
        raise RuntimeError(f"images.txt not found at {images_txt}")

    image_names = read_image_names_from_colmap(images_txt)
    if not image_names:
        raise RuntimeError(f"No image names parsed from {images_txt}")

    n = min(len(render_files), len(image_names))
    if len(render_files) != len(image_names):
        print(f"[WARN] #renders={len(render_files)} != #colmap_images={len(image_names)}")
        print("       将按前 n=min(...) 做一一对应。")

    psnrs = []
    used = 0

    for i in range(n):
        r_path = render_files[i]
        img_name = image_names[i]

        c_path = find_clean_image(clean_images_dir, img_name)
        if c_path is None:
            print(f"[WARN] Clean image not found for {img_name}, skip")
            continue

        r = load_image(r_path)
        H, W = r.shape[:2]
        c = load_image(c_path, size=(W, H))

        if c.shape != r.shape:
            print(f"[WARN] Shape mismatch for {img_name}, skip")
            continue

        p = psnr(r, c)
        psnrs.append(p)
        used += 1

    if not psnrs:
        print("No valid pairs for PSNR.")
        return

    mean_psnr = sum(psnrs) / len(psnrs)
    print(f"Num images used: {used}")
    print(f"Mean PSNR vs CLEAN (COLMAP-based): {mean_psnr:.3f} dB")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python render_then_eval.py <renders_dir> <clean_images_dir> <sparse_dir>")
        sys.exit(1)

    renders_dir = sys.argv[1]
    clean_images_dir = sys.argv[2]
    sparse_dir = sys.argv[3]

    eval_vs_clean_colmap(renders_dir, clean_images_dir, sparse_dir)
