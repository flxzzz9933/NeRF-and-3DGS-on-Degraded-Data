# NeRF-and-3DGS-on-Degraded-Data# NeRF & 3DGS on Degraded Data

This repo contains my experiments on how image degradations affect:
- **NeRF (nerfacto, Nerfstudio)**
- **3D Gaussian Splatting (3DGS)**

I run controlled degradations on:
- **Tanks & Temples – Truck**
- **NeRF Synthetic – chair, drums**

For each setting I train NeRF / 3DGS and evaluate against **clean ground truth** using **PSNR / SSIM / LPIPS**.

---

## 1. Repo layout (high level)

```text
.
├── 3dgs/                     # 3DGS helpers + 3dgs results
│   ├── add_pose_noise_colmap.py
│   ├── eval_metrics_vs_clean.py
│   ├── eval_psnr_vs_clean_colmap.py
│   ├── filter_sparse_images_to_existing.py
│   ├── fix_colmap_cameras_pinhole.py
│   ├── json_to_colmap_v3.py
│   ├── gaussian-splatting -> /home/flx/gaussian-splatting (local clone)
│   └── results/
│       ├── gs_eval_all.json          # main 3DGS metrics (Truck + nerf_synthetic)
│       ├── nerf_synthetic/           # chair/drums metrics text
│       └── truck/                    # Truck T0–T2 raw eval logs
├── nerf/ nsenv/               # Nerfstudio virtualenv (local only, not needed to clone)
├── nerf_data/                 # Datasets (clean + degraded)
│   ├── nerf_synthetic/        # Original Blender data
│   ├── nerf_synthetic_degraded/
│   ├── tanks/                 # Tanks & Temples Truck (COLMAP-style)
│   └── tanks_degraded/        # T0/T1/T2 degraded Truck
├── nerf_projects/
│   ├── nerf_synthetic/        # Nerfstudio runs for chair/drums E0–E8
│   │   ├── chair/
│   │   ├── drums/
│   │   ├── collect_nerf_eval_all.py
│   │   └── nerf_eval_all.json # All NeRF metrics (chair + drums)
│   └── tanks_degraded/Truck   # Nerfstudio runs for Truck T0/T1/T2
├── scripts (root)
│   ├── eval_clean_synthetic.sh           # Nerfstudio eval vs clean GT
│   ├── fix_paths.py                      # Fix absolute paths inside json, etc.
│   ├── fix_rgba_all.py                   # Fix RGBA images (alpha handling)
│   ├── make_transform_all.py             # Build transforms for degraded splits
│   ├── prepare_llff_degraded.py          # Make degraded LLFF data (for future use)
│   ├── prepare_nerf_synthetic_degraded.py# Make E0–E8 for chair/drums
│   ├── prepare_tanks_degraded.py         # Make Truck T0–T2
│   ├── run_3dgs_nerfsynthetic_render_eval.sh
│   ├── run_experiments.py                # Main NeRF training launcher
│   ├── update_eval_jsons.py              # Merge / update JSON metrics files
│   ├── nerf_data_index.json              # Index of all scenes / experiments
│   ├── nerf_results.csv                  # Summary table for NeRF
│   └── .gitignore
