#!/bin/bash
# Sequential GPU pass: (1) Llama conf30 DPO, (2) Qwen component-ablation
# variants + reweighted-SFT baselines. Swaps the local-pipeline config to
# the Qwen backbone for step 2 and restores it afterwards (trap-guarded).
set -euo pipefail
cd "$(dirname "$0")/../.."

CFG=configs/local_pipeline.yaml
PY=venv/bin/python

restore_config() {
  sed -i '' 's|^  id: Qwen/Qwen2.5-3B-Instruct$|  id: meta-llama/Llama-3.2-3B-Instruct|' "$CFG"
  sed -i '' 's|^  shortname: qwen25_3b$|  shortname: llama32_3b|' "$CFG"
  $PY -c "import sys; sys.path.insert(0,'.'); from scripts.local_pipeline import config_hash; open('results/local_pipeline/config_hash.txt','w').write(config_hash()+'\n')"
}
trap restore_config EXIT

echo "=== [1/2] Llama conf30_dpo ==="
$PY scripts/local_pipeline/stage_05_dpo_variants.py --only conf30_dpo
$PY scripts/local_pipeline/stage_06_eval_oof.py --variants conf30_dpo

echo "=== [2/2] Qwen pass: swap config ==="
sed -i '' 's|^  id: meta-llama/Llama-3.2-3B-Instruct$|  id: Qwen/Qwen2.5-3B-Instruct|' "$CFG"
sed -i '' 's|^  shortname: llama32_3b$|  shortname: qwen25_3b|' "$CFG"

$PY scripts/local_pipeline/stage_05_dpo_variants.py --only agree30_dpo agree30_tb2_dpo uncert30_dpo conf30_dpo
$PY scripts/arr_ablations/train_reweighted_sft_local.py
$PY scripts/local_pipeline/stage_06_eval_oof.py --variants agree30_dpo agree30_tb2_dpo uncert30_dpo conf30_dpo wsft softlabel_sft
$PY scripts/local_pipeline/stage_07_unified_tables.py

echo "=== restoring Llama config ==="
restore_config
trap - EXIT
$PY scripts/local_pipeline/stage_07_unified_tables.py
echo "QWEN PASS COMPLETE"
