# LOCAL_PIPELINE_RULES.md

Rules of the road for the **local-model pipeline** (Llama-3.1-8B / Mistral-7B on Apple Silicon).

The goal of this document is zero-drift: every stage has checkable invariants, outputs go to exactly one place, and nothing is lost between reruns. Before every stage runs, and after every stage finishes, the validator (`scripts/local_pipeline/validate.py`) MUST be green.

---

## 0 · Why a local pipeline alongside the gpt-4o one

The `gpt-4o` track is the main experimental track. The local track is a **comparability mirror**: same test set, same prompt, same R(x) formula, same OOF weight fitting. Its job is to show that the RA-DPO / Smart-sampling story transfers off OpenAI hosted models.

Do not "replace" gpt-4o with local; both tracks coexist. Every local result file name ends with `_local` to make it impossible to confuse with the gpt-4o outputs.

---

## 1 · Model choice

Primary: **`meta-llama/Llama-3.1-8B-Instruct`**.
- Good multilingual coverage (EN + ES — EXIST has both).
- Large TRL/PEFT community, many DPO recipes.
- ~16 GB in fp16; fits comfortably on a 64 GB M4 Max with room for LoRA state.

Alternative (cited in the comparison papers): `mistralai/Mistral-7B-Instruct-v0.3`.
Fast iteration (only): `meta-llama/Llama-3.2-3B-Instruct`.

The choice is fixed in `configs/local_pipeline.yaml` → `model.id`. Do **not** override ad-hoc at the command line; change the config and re-run the validator instead.

Sources for this choice (April 2026):
- [Best Ways to Run LLM Locally on Mac — DEV Community](https://dev.to/mehmetakar/5-ways-to-run-llm-locally-on-mac-cck)
- [Best Local LLMs on Apple Silicon](https://apxml.com/posts/best-local-llm-apple-silicon-mac)
- [Best Open-Source LLMs in 2026](https://mljourney.com/best-open-source-llms-in-2026-a-practical-guide-by-use-case/)

---

## 2 · Hardware contract (M4 Max, 64 GB)

- `torch.backends.mps.is_available()` must be `True`. If not, stop.
- **`bf16` is NOT supported on MPS**. Use `fp16`. Config forbids `bf16` automatically.
- **`bitsandbytes` 4-bit is NOT available on MPS.** We train in fp16; we do NOT try to quantise at training time.
- Memory budget per stage (approx):
  - 8B fp16 forward: ~17 GB
  - 8B + LoRA (r=16) + optimizer: ~24 GB
  - Keep batch size ≤ 4 for 8B; ≤ 8 for 3B.

The preflight script runs a 10-token generation sanity check and aborts if any of these conditions fail.

---

## 3 · Invariants that must hold across EVERY rerun

These are checked by `validate.py` and cannot be skipped.

| # | Invariant | Failure means |
|---|---|---|
| I1 | Test set size is **exactly 692 samples** | wrong data split |
| I2 | Per-instance arrays (predictions, confidences, agreements, sigmoid_scores, correct) all have **length 692** | alignment bug |
| I3 | `agreements` for the local run **equals** the gpt-4o pipeline's `agreements` (bit-for-bit) | data-loader mismatch |
| I4 | `sigmoid_scores` for the local run **equals** the gpt-4o pipeline's `sigmoid_scores` (tweet-level, model-independent) | token-scoring drift |
| I5 | Every `confidence` value is in `[0, 1]` | logprob bug |
| I6 | Every `agreement` value is in `{0.5, 0.667, 0.833, 1.0}` | majority_vote broken |
| I7 | `predictions` values are a subset of `{"YES", "NO"}` | prompt/parse bug |
| I8 | Prompt strategy used for ALL fine-tuning / coverage-accuracy rows is **"structured"** | prompt drift |
| I9 | DPO training JSONL file sizes match `configs/local_pipeline.yaml → data.training_pairs` | wrong subset loaded |
| I10 | α + β + γ = 1.0 (OOF mean) within 1e-6 | normalization bug |
| I11 | Coverage-accuracy @100% == `standard_metrics.accuracy` within 1e-6 | R(x) ranking bug |
| I12 | Every output JSON has the keys: `model`, `model_id`, `training_pairs`, `standard_metrics`, `per_instance`, `n_samples`, `timestamp`, `prompt_strategy` | schema drift |

Any failed invariant aborts the pipeline with a non-zero exit code.

---

## 4 · Directory layout (canonical)

```
results/
  final_reliability_3factor/          # gpt-4o per-instance files (DO NOT TOUCH from local stages)
  unified_gpt4o/                      # gpt-4o tables
  local_pipeline/                     # everything produced by local pipeline
    config_hash.txt                   # hash of the config used — validator compares
    per_instance/                     # one JSON per model variant, schema identical to final_reliability_3factor/
      llama31_8b_base_local.json
      llama31_8b_sft_local.json
      llama31_8b_std_dpo_local.json
      ...
    unified/                          # parallel of results/unified_gpt4o/
      fine_tuning.csv
      coverage_accuracy.csv
      weights.csv
      summary.json
    training/                         # LoRA adapter checkpoints per variant
      sft/...
      std_dpo/...
      smart30_dpo/...
      ra_dpo/...
    logs/                             # per-stage stdout/stderr, wall-clock, memory
    checkpoints.json                  # last completed stage per variant
models/
  local_pipeline/<model-shortname>/   # downloaded base weights (cached)
```

**Rule:** nothing local-pipeline-related is written outside `results/local_pipeline/`, `models/local_pipeline/`, or `scripts/local_pipeline/`.

---

## 5 · Canonical run order

Each stage writes a JSON under `results/local_pipeline/logs/` indicating status. `run_all.py` reads `checkpoints.json` to skip completed stages.

```
01_preflight          → checks hardware, downloads weights
02_baseline_inference → test-set predictions + confidences with structured prompt
03_reuse_sigmoid      → copy sigmoid_scores and agreements from gpt-4o pipeline (invariants I3, I4)
04_sft_lora           → SFT on train split (5,535 SFT pairs)
05_dpo_variants       → DPO for: Std, Smart-10/30/50, Random-50, Ambiguous-only, RA-DPO
06_eval_all           → run each fine-tuned variant on the test set
07_oof_rx             → 5-fold OOF α/β/γ and coverage-accuracy per variant
08_unified_tables     → build parallel unified tables
09_validate_final     → run the full validator suite one last time
```

Stage `N` MUST NOT run until stage `N-1`'s invariants pass.

---

## 6 · Prompt / label rules (non-negotiable)

- The structured prompt for inference comes from `ra_dpo/pipeline/prompts.py → get_system_prompt("structured", lang)`. Do not inline it into local scripts.
- For SFT, training data is `results/openai_sft_train.jsonl` (5,535 lines, unchanged).
- For DPO, training data is `results/smart_sampling/*.jsonl` (same files used by the gpt-4o track).
- Majority vote uses `ra_dpo.data.data_loader.majority_vote`. Ties (3/3) are broken by whichever label lexicographically comes first — **this is a property of the shared data loader** and must match the gpt-4o track.

---

## 7 · Hyperparameters (locked, tracked by config hash)

LoRA for both SFT and DPO on 7B / 8B:
```
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

SFT:
```
epochs = 1
lr = 2e-5
batch_size = 2  (per_device) + grad_accum = 8   (effective 16)
max_seq_len = 512
```

DPO:
```
beta = 0.1
epochs = 1
lr = 5e-6
batch_size = 1 + grad_accum = 16   (effective 16)
max_prompt_length = 384
max_length = 512
```

These are mirrored from the gpt-4o job settings where possible. Changing any value bumps `config_hash.txt`; the validator compares it against the hash stored in every per-instance JSON.

Hash history: `0a789ba9038fff85` (original 3B runs, all 18 per-instance files
for llama32_3b + qwen25_3b) → `a4974452b5f7d74b` (2026-07-08, ARR revision:
added component-ablation variants `agree30_dpo`, `agree30_tb2_dpo`,
`uncert30_dpo`, `conf30_dpo` and reweighted-SFT baselines `wsft`,
`softlabel_sft` to `training_pairs`; corrected `sft` count to 5536). Hash-drift
warnings on the 18 pre-existing per-instance files are expected and benign —
no hyperparameter changed, only new variant registrations.

---

## 8 · What to do when something looks wrong

1. Run `python scripts/local_pipeline/validate.py --strict`. Read which invariant failed.
2. Fix the failing stage, rerun from that stage only. Do NOT nuke the whole `results/local_pipeline/` tree.
3. If an invariant fix requires a data-loader change, the validator's invariants I3 and I4 will catch any drift in `agreement` or `sigmoid_scores` bit-for-bit — fix the upstream cause, then rerun.
4. If MPS runs out of memory, reduce `per_device` batch size (doubling `grad_accum`), not sequence length.
5. If a LoRA run diverges, the first thing to check is whether `attention_mask` is being set for all padded tokens. MPS is silent about this.

---

## 9 · Comparability with gpt-4o

Exactly three things change between the gpt-4o track and the local track:
1. The model producing predictions.
2. The model producing confidences (logprob of the first answer token).
3. The trained adapters / fine-tuned weights.

Everything else (test split, prompt, sigmoid scoring, agreement predictor, OOF weight fitting, coverage thresholds) is shared code paths. This is what makes the comparison meaningful.

---

## 10 · Validator checklist (what `validate.py` checks in one shot)

- Hardware: MPS available, PyTorch ≥ 2.1, transformers ≥ 4.42, trl ≥ 0.9, peft ≥ 0.11
- Config: `configs/local_pipeline.yaml` parses; hash matches `results/local_pipeline/config_hash.txt`.
- Data: `EXIST2023_training.json` present; `create_train_val_test_split(df)` yields 5,536 / 692 / 692.
- Invariants I1–I12 on every per-instance JSON under `results/local_pipeline/per_instance/`.
- Schema: expected keys present; no NaNs in confidence / agreement / sigmoid arrays.
- Cross-reference: `agreements` and `sigmoid_scores` bit-match `results/final_reliability_3factor/gpt-4o_base.json`.
- Tables: `fine_tuning.csv`, `coverage_accuracy.csv`, `weights.csv` exist; numbers are in-range.
- Checkpoints: every fine-tuned variant has a LoRA adapter under `training/`.

The validator exits 0 if everything is good, non-zero with a human-readable error if anything fails.

---

*This file is the source of truth. If a script contradicts this file, the file wins.*
