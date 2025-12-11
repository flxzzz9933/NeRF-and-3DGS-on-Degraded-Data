#!/usr/bin/env bash
set -euo pipefail

# 干净 GT 的根目录
CLEAN_ROOT="/home/flx/nerf_data/nerf_synthetic"

SCENES="chair drums"
EXPS="E0_C0_I0_P0 E1_C1_I0_P0 E2_C2_I0_P0 E3_C0_I1_P0 E4_C0_I2_P0 E5_C0_I0_P1 E6_C0_I0_P2 E7_C1_I1_P0 E8_C0_I1_P1"

for SCENE in $SCENES; do
  for EXP in $EXPS; do
    BASE="$HOME/nerf_projects/nerf_synthetic/$SCENE/$EXP/$SCENE/nerfacto"
    [ ! -d "$BASE" ] && continue

    # 找到最新一次 run（排除 *_old）
    RUN_DIR=$(find "$BASE" -maxdepth 1 -type d -name "2025-12-10*" ! -name "*_old" | sort | tail -n 1)
    [ -z "$RUN_DIR" ] && continue

    CONFIG="$RUN_DIR/config.yml"
    [ ! -f "$CONFIG" ] && continue

    CLEAN_CFG="$RUN_DIR/config_clean.yml"
    CLEAN_JSON="$RUN_DIR/eval_clean.json"
    CLEAN_DATA="$CLEAN_ROOT/$SCENE"

    echo "------------------------------------------------------------"
    echo "Using Clean GT root: $CLEAN_DATA"
    echo "Scene/Exp:          $SCENE  $EXP"
    echo "Run dir:            $RUN_DIR"
    echo "Clean config:       $CLEAN_CFG"
    echo "Clean metrics out:  $CLEAN_JSON"

    # 纯文本方式：在 dataparser block 里把 data: 那一行改成干净 GT
    python - << PY
from pathlib import Path

cfg_path = Path(r"$CONFIG")
new_data = r"$CLEAN_DATA"

text = cfg_path.read_text().splitlines()

out_lines = []
seen_dataparser = False
replaced = False
dataparser_indent = None

for line in text:
    stripped = line.lstrip()
    indent = len(line) - len(stripped)

    # 进入 dataparser: 这个 block
    if "dataparser" in stripped.split()[0]:
        seen_dataparser = True
        dataparser_indent = indent
        out_lines.append(line)
        continue

    # 在 dataparser block 里，寻找第一行 data:
    if seen_dataparser and not replaced:
        # 如果缩进缩回到 dataparser 同级，说明离开 dataparser block 了
        if stripped and indent <= dataparser_indent:
            seen_dataparser = False

        if stripped.startswith("data:"):
            # 保留原来的缩进，只改路径
            out_lines.append(" " * indent + f"data: {new_data}")
            replaced = True
            continue

    out_lines.append(line)

out_path = cfg_path.with_name("config_clean.yml")
out_path.write_text("\n".join(out_lines) + "\n")

if not replaced:
    print(f"[WARN] 在 {cfg_path} 里没找到 dataparser 下的 data: 行，没有做修改。")
PY

    # 用改好的 config 跑 eval（只加载权重）
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 ns-eval \
      --load-config "$CLEAN_CFG" \
      --output-path "$CLEAN_JSON"

    echo "Done CLEAN-GT eval for $SCENE $EXP"
  done
done

echo "全部 CLEAN-GT eval 跑完了。"
