import json
from pathlib import Path

BASE = Path.home() / "nerf_projects" / "nerf_synthetic"

# 目前有的 scene：chair, drums, truck
SCENES = ["chair", "drums", "truck"]


def latest_run(run_root: Path) -> Path | None:
    """在某个 nerfacto 根目录下找最新的 run 目录。"""
    if not run_root.exists():
        return None
    runs = sorted([p for p in run_root.iterdir() if p.is_dir()], reverse=True)
    return runs[0] if runs else None


def collect_for_scene(scene: str) -> dict:
    """
    scene 目录结构假设为：
    BASE/scene/EXP_NAME/scene/nerfacto/<timestamp>/
    例如：nerf_synthetic/truck/t1_C0_I1_P0/truck/nerfacto/2025-12-10_xxxx
    """
    scene_root = BASE / scene
    if not scene_root.exists():
        return {}

    result: dict[str, dict] = {}

    for exp_dir in sorted([p for p in scene_root.iterdir() if p.is_dir()]):
        exp_name = exp_dir.name

        run_root = exp_dir / scene / "nerfacto"
        run_dir = latest_run(run_root)
        if run_dir is None:
            continue

        metrics_entry: dict[str, dict] = {}

        # 降质数据 eval（标准命名）
        degraded = run_dir / "eval_metrics.json"
        if degraded.exists():
            with degraded.open("r") as f:
                metrics_entry["degraded"] = json.load(f)

        # clean 数据 eval，兼容两种可能命名：eval_clean.json / eval_metrics_clean.json
        clean_candidates = [
            run_dir / "eval_clean.json",
            run_dir / "eval_metrics_clean.json",
        ]
        for cp in clean_candidates:
            if cp.exists():
                with cp.open("r") as f:
                    metrics_entry["clean"] = json.load(f)
                break

        if metrics_entry:
            result[exp_name] = {
                "run_dir": str(run_dir),
                "metrics": metrics_entry,
            }

    return result


def main():
    all_results = {}
    for scene in SCENES:
        all_results[scene] = collect_for_scene(scene)

    out_path = BASE / "nerf_eval_all.json"
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved combined eval metrics to: {out_path}")


if __name__ == "__main__":
    main()
