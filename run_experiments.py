#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import subprocess
import time


def parse_last_psnr(log_path: str):
    """从训练 log 里粗暴地找最后一个 psnr 数字。"""
    if not os.path.exists(log_path):
        return None
    psnr = None
    pat = re.compile(r"psnr[^0-9]*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                try:
                    psnr = float(m.group(1))
                except ValueError:
                    continue
    return psnr


def log_has_training_finished(log_path: str) -> bool:
    """检查 log 里是不是已经出现过 'Training Finished'。"""
    if not os.path.exists(log_path):
        return False
    try:
        size = os.path.getsize(log_path)
        read_size = 8192
        with open(log_path, "rb") as f:
            if size > read_size:
                f.seek(size - read_size)
            tail = f.read().decode("utf-8", errors="ignore")
        return "Training Finished" in tail
    except OSError:
        return False


def load_entries(index_path):
    """兼容 list 和 dict 两种 index 格式。"""
    with open(index_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "entries" in data and isinstance(data["entries"], list):
            return data["entries"]
        return list(data.values())

    raise TypeError(f"Unknown index format in {index_path}: {type(data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="nerf_data_index.json")
    parser.add_argument(
        "--root",
        default="nerf_data",
        help="数据根目录，index 里的 data_dir 是相对这个路径的。",
    )
    parser.add_argument(
        "--out-root",
        default="nerf_projects",
        help="ns-train 的输出和 log 存这里。",
    )
    parser.add_argument(
        "--method",
        default="nerfacto",
        help="Nerfstudio 方法名，比如 nerfacto 或 splatfacto。",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=20000,
        help="ns-train 的最大迭代步数。",
    )
    parser.add_argument("--results-csv", default="nerf_results.csv")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印命令，不真正跑。",
    )
    parser.add_argument(
        "--only-dataset",
        type=lambda s: s.split(","),
        default=None,
        help="只跑这些 dataset，逗号分隔，比如 'llff_degraded,nerf_synthetic_degraded'。",
    )
    parser.add_argument(
        "--only-scene",
        type=lambda s: s.split(","),
        default=None,
        help="只跑这些 scene，比如 'fern,room,chair,drums,Truck'。",
    )
    args = parser.parse_args()

    entries = load_entries(args.index)
    os.makedirs(args.out_root, exist_ok=True)

    # 已有结果（根据 run_name）支持断点续跑
    already_done = set()
    if os.path.exists(args.results_csv):
        with open(args.results_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rn = row.get("run_name")
                if rn:
                    already_done.add(rn)

    fieldnames = [
        "run_name",
        "dataset",
        "scene",
        "experiment",
        "method",
        "data_dir",
        "output_dir",
        "max_iters",
        "train_seconds",
        "final_psnr",
    ]
    csv_exists = os.path.exists(args.results_csv)
    csv_file = open(args.results_csv, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()

    try:
        for e in entries:
            if args.only_dataset and e["dataset"] not in args.only_dataset:
                continue
            if args.only_scene and e["scene"] not in args.only_scene:
                continue

            run_name = f"{e['dataset']}_{e['scene']}_{e['experiment']}"
            data_dir = os.path.join(args.root, e["data_dir"])
            out_dir = os.path.join(
                args.out_root, e["dataset"], e["scene"], e["experiment"]
            )
            os.makedirs(out_dir, exist_ok=True)
            log_path = os.path.join(out_dir, "train.log")

            # 1) CSV 里已经有了
            if run_name in already_done:
                print(f"[skip] {run_name} already in {args.results_csv}")
                continue

            # 2) 有完整 log（手动跑过），只写 CSV 不重训
            if log_has_training_finished(log_path):
                print(f"[reuse] {run_name}: found finished log, only write CSV")
                psnr = parse_last_psnr(log_path)
                writer.writerow(
                    dict(
                        run_name=run_name,
                        dataset=e["dataset"],
                        scene=e["scene"],
                        experiment=e["experiment"],
                        method=args.method,
                        data_dir=data_dir,
                        output_dir=out_dir,
                        max_iters=args.max_iters,
                        train_seconds="",
                        final_psnr=psnr if psnr is not None else "",
                    )
                )
                csv_file.flush()
                continue

            # --- ✅ 唯一修改的部分：cmd ---
            cmd = [
                "ns-train",
                args.method,
                "--data", data_dir,
                "--output-dir", out_dir,
                "--max-num-iterations", str(args.max_iters),
                "--viewer.quit-on-train-completion", "True",
            ]
            # ----------------------------------

            print(f"\n=== Running {run_name} ===")
            print(" ".join(cmd))

            if args.dry_run:
                continue

            start = time.time()
            with open(log_path, "w") as lf:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            end = time.time()
            train_secs = round(end - start, 2)

            if proc.returncode != 0:
                print(
                    f"[warn] {run_name} failed with code {proc.returncode}, "
                    f"see {log_path}"
                )
                writer.writerow(
                    dict(
                        run_name=run_name,
                        dataset=e["dataset"],
                        scene=e["scene"],
                        experiment=e["experiment"],
                        method=args.method,
                        data_dir=data_dir,
                        output_dir=out_dir,
                        max_iters=args.max_iters,
                        train_seconds=train_secs,
                        final_psnr="",
                    )
                )
                csv_file.flush()
                continue

            psnr = parse_last_psnr(log_path)
            writer.writerow(
                dict(
                    run_name=run_name,
                    dataset=e["dataset"],
                    scene=e["scene"],
                    experiment=e["experiment"],
                    method=args.method,
                    data_dir=data_dir,
                    output_dir=out_dir,
                    max_iters=args.max_iters,
                    train_seconds=train_secs,
                    final_psnr=psnr if psnr is not None else "",
                )
            )
            csv_file.flush()
            print(f"[done] {run_name}  time={train_secs}s  psnr={psnr}")

    finally:
        csv_file.close()


if __name__ == "__main__":
    main()
