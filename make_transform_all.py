import os
import json
import math
from PIL import Image

ROOT = "nerf_data/nerf_synthetic_degraded"

def compute_intrinsics_for_scene(scene_root, scene_name):
    """
    对某个 scene（比如 chair / drums）：
    用 E0_C0_I0_P0 的 transforms_train.json + 一张图像
    计算 intrinsics。
    """
    clean_dir = os.path.join(scene_root, "E0_C0_I0_P0", scene_name)
    clean_json = os.path.join(clean_dir, "transforms_train.json")

    if not os.path.exists(clean_json):
        raise FileNotFoundError(f"[{scene_name}] Clean E0 transforms_train.json not found: {clean_json}")

    with open(clean_json, "r") as f:
        clean = json.load(f)

    camera_angle_x = clean["camera_angle_x"]

    # 用 clean 的一张图片推导 w/h
    sample_frame = clean["frames"][0]
    sample_img_path = os.path.join(clean_dir, sample_frame["file_path"])
    # 有的 transforms 不带后缀，我们兜底一下
    if not (sample_img_path.endswith(".png") or sample_img_path.endswith(".jpg")):
        if os.path.exists(sample_img_path + ".png"):
            sample_img_path += ".png"
        elif os.path.exists(sample_img_path + ".jpg"):
            sample_img_path += ".jpg"

    if not os.path.exists(sample_img_path):
        raise FileNotFoundError(f"[{scene_name}] Sample image not found: {sample_img_path}")

    w, h = Image.open(sample_img_path).size

    fl_x = 0.5 * w / math.tan(0.5 * camera_angle_x)
    fl_y = fl_x
    cx = w * 0.5
    cy = h * 0.5
    camera_angle_y = 2 * math.atan(h / (2 * fl_y))

    intrinsics = {
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "w": w,
        "h": h,
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
    }

    print(f"[INFO] {scene_name} intrinsics:\n{intrinsics}")
    return intrinsics


def merge_transforms(split_dir, intrinsics):
    """读取 transforms_train/val/test.json 并合并为一个 transforms.json"""
    t_train = os.path.join(split_dir, "transforms_train.json")
    t_val   = os.path.join(split_dir, "transforms_val.json")
    t_test  = os.path.join(split_dir, "transforms_test.json")

    if not (os.path.exists(t_train) and os.path.exists(t_val) and os.path.exists(t_test)):
        return None

    with open(t_train, "r") as f:
        train = json.load(f)
    with open(t_val, "r") as f:
        val = json.load(f)
    with open(t_test, "r") as f:
        test = json.load(f)

    merged = {
        **intrinsics,
        "frames": []
    }

    for fr in train["frames"]:
        fr["split"] = "train"
        merged["frames"].append(fr)

    for fr in val["frames"]:
        fr["split"] = "val"
        merged["frames"].append(fr)

    for fr in test["frames"]:
        fr["split"] = "test"
        merged["frames"].append(fr)

    return merged


def main():
    total_updated = 0

    # 遍历所有 scene（chair、drums、lego...）
    for scene_name in sorted(os.listdir(ROOT)):
        scene_root = os.path.join(ROOT, scene_name)
        if not os.path.isdir(scene_root):
            continue

        print(f"\n============================")
        print(f"[SCENE] {scene_name}")
        print(f"============================")

        try:
            intrinsics = compute_intrinsics_for_scene(scene_root, scene_name)
        except FileNotFoundError as e:
            print(f"[SKIP SCENE] {e}")
            continue

        updated_scene = 0

        # 每个 scene 下面是 E*_C*_I*_P* 这些实验目录
        for exp in sorted(os.listdir(scene_root)):
            exp_dir = os.path.join(scene_root, exp, scene_name)
            if not os.path.isdir(exp_dir):
                continue

            merged = merge_transforms(exp_dir, intrinsics)
            if merged is None:
                print(f"[SKIP] No transforms_train/val/test found for {scene_name}/{exp}")
                continue

            out_path = os.path.join(exp_dir, "transforms.json")
            with open(out_path, "w") as f:
                json.dump(merged, f, indent=2)

            print(f"[OK] Updated transforms.json for {scene_name}/{exp}")
            updated_scene += 1
            total_updated += 1

        print(f"[SCENE DONE] {scene_name}: {updated_scene} experiments updated")

    print(f"\n[DONE] Total transforms.json updated: {total_updated}")


if __name__ == "__main__":
    main()
