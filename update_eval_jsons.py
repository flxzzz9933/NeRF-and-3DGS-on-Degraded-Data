#!/usr/bin/env python3
import json
import re
from pathlib import Path

HOME = Path.home()

# 现有的 nerf 指标总表
NERF_JSON = HOME / "nerf_projects/nerf_synthetic/nerf_eval_all.json"

# 刚才保存的 truck 指标文本
TRUCK_METRICS_TXT = HOME / "3dgs/results/truck_eval_vs_clean.txt"

# 要新写出的 3DGS 总表
GS_JSON = HOME / "3dgs/results/gs_eval_all.json"


def parse_truck_metrics(text: str):
    """
    解析 truck_eval_vs_clean.txt 里面 t0/t1/t2 的三组指标
    返回类似：
    {
      "T0": {"psnr": 9.863, "ssim": 0.3770, "lpips": 0.7110},
      ...
    }
    """
    blocks = re.split(r"\n\s*\n", text.strip())
    out = {}
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        tag = lines[0].lower()
        if tag not in ("t0", "t1", "t2"):
            continue
        key = tag.upper()
        m_dict = {}
        for line in lines[1:]:
            m = re.match(r"Mean\s+(\w+):\s*([0-9.]+)", line)
            if not m:
                continue
            metric = m.group(1).lower()   # psnr / ssim / lpips
            val = float(m.group(2))
            m_dict[metric] = val
        out[key] = m_dict
    return out


def update_nerf_json_with_truck(truck_metrics):
    """把 truck 的三组指标写进 nerf_eval_all.json 里的 'truck' 字段"""

    print(f"[INFO] Loading {NERF_JSON}")
    with NERF_JSON.open("r") as f:
        data = json.load(f)

    truck_block = {}
    for t_key, m in truck_metrics.items():
        run_dir = str(HOME / "3dgs/results/truck" / t_key)
        truck_block[t_key] = {
            "run_dir": run_dir,
            "metrics": {
                # 我们只有 vs clean 的 eval，就放在 'clean' 里，结构跟 nerf 的 clean 分支一样
                "clean": {
                    "experiment_name": "truck",
                    "method_name": "3dgs",
                    "checkpoint": "",  # 3dgs 没有 ckpt 就留空
                    "results": {
                        "psnr": m.get("psnr"),
                        "ssim": m.get("ssim"),
                        "lpips": m.get("lpips"),
                    },
                }
            },
        }

    data["truck"] = truck_block

    with NERF_JSON.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[INFO] Updated {NERF_JSON} with truck metrics.")


def build_gs_eval_json(truck_metrics):
    """
    把 3DGS 的结果写成一个新的 gs_eval_all.json：
    顶层: chair / drums / truck
    每个下面是 E0...E8 或 T0...T2
    """

    # 这些是你刚刚贴出来的 3DGS nerf_synthetic 指标（VS clean）
    gs_nerf_metrics = {
        "chair": {
            "E0_C0_I0_P0": {"psnr": 32.142, "ssim": 0.9842, "lpips": 0.0342},
            "E1_C1_I0_P0": {"psnr": 31.992, "ssim": 0.9834, "lpips": 0.0347},
            "E2_C2_I0_P0": {"psnr": 32.028, "ssim": 0.9837, "lpips": 0.0354},
            "E3_C0_I1_P0": {"psnr": 31.700, "ssim": 0.9798, "lpips": 0.0384},
            "E4_C0_I2_P0": {"psnr": 30.491, "ssim": 0.9641, "lpips": 0.0546},
            "E5_C0_I0_P1": {"psnr": 22.624, "ssim": 0.8832, "lpips": 0.0750},
            "E6_C0_I0_P2": {"psnr": 17.708, "ssim": 0.8280, "lpips": 0.1269},
            "E7_C1_I1_P0": {"psnr": 31.519, "ssim": 0.9798, "lpips": 0.0372},
            "E8_C0_I1_P1": {"psnr": 23.202, "ssim": 0.8914, "lpips": 0.0723},
        },
        "drums": {
            "E0_C0_I0_P0": {"psnr": 25.776, "ssim": 0.9481, "lpips": 0.0814},
            "E1_C1_I0_P0": {"psnr": 25.568, "ssim": 0.9456, "lpips": 0.0818},
            "E2_C2_I0_P0": {"psnr": 25.862, "ssim": 0.9481, "lpips": 0.0797},
            "E3_C0_I1_P0": {"psnr": 25.784, "ssim": 0.9470, "lpips": 0.0840},
            "E4_C0_I2_P0": {"psnr": 25.583, "ssim": 0.9418, "lpips": 0.0912},
            "E5_C0_I0_P1": {"psnr": 20.561, "ssim": 0.8568, "lpips": 0.1200},
            "E6_C0_I0_P2": {"psnr": 16.427, "ssim": 0.7779, "lpips": 0.1879},
            "E7_C1_I1_P0": {"psnr": 25.595, "ssim": 0.9449, "lpips": 0.0831},
            "E8_C0_I1_P1": {"psnr": 20.772, "ssim": 0.8592, "lpips": 0.1178},
        },
    }

    gs_data = {}

    # 填 chair / drums
    for scene, exps in gs_nerf_metrics.items():
        scene_dict = {}
        for exp, m in exps.items():
            run_dir = str(HOME / "3dgs/results/nerf_synthetic" / scene / exp)
            scene_dict[exp] = {
                "run_dir": run_dir,
                "metrics": {
                    "clean": {
                        "experiment_name": scene,
                        "method_name": "3dgs",
                        "checkpoint": "",
                        "results": {
                            "psnr": m["psnr"],
                            "ssim": m["ssim"],
                            "lpips": m["lpips"],
                        },
                    }
                },
            }
        gs_data[scene] = scene_dict

    # 填 truck T0/T1/T2
    truck_dict = {}
    for t_key, m in truck_metrics.items():
        run_dir = str(HOME / "3dgs/results/truck" / t_key)
        truck_dict[t_key] = {
            "run_dir": run_dir,
            "metrics": {
                "clean": {
                    "experiment_name": "truck",
                    "method_name": "3dgs",
                    "checkpoint": "",
                    "results": {
                        "psnr": m.get("psnr"),
                        "ssim": m.get("ssim"),
                        "lpips": m.get("lpips"),
                    },
                }
            },
        }
    gs_data["truck"] = truck_dict

    with GS_JSON.open("w") as f:
        json.dump(gs_data, f, indent=2)
    print(f"[INFO] Wrote 3DGS metrics to {GS_JSON}")


def main():
    if not TRUCK_METRICS_TXT.exists():
        raise SystemExit(f"[ERROR] {TRUCK_METRICS_TXT} not found, "
                         "请先按说明创建 truck_eval_vs_clean.txt")

    txt = TRUCK_METRICS_TXT.read_text()
    truck_metrics = parse_truck_metrics(txt)

    if not truck_metrics:
        raise SystemExit("[ERROR] 没在 truck_eval_vs_clean.txt 里解析到任何 t0/t1/t2 指标，"
                         "检查一下格式是不是跟示例一样。")

    update_nerf_json_with_truck(truck_metrics)
    build_gs_eval_json(truck_metrics)


if __name__ == "__main__":
    main()
