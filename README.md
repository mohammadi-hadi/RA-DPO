# RA-DPO

Code for the paper *RA-DPO: Reliability-Aware Preference Optimization and Selective Prediction for Sexism Detection* (under review).

RA-DPO scores every instance with a reliability function

```
R(x) = alpha * confidence + beta * annotator_agreement + gamma * (1 - token_uncertainty)
```

whose weights are fit by out-of-fold logistic regression, and uses it two ways: (1) weighting preference pairs during DPO training, and (2) selective prediction at inference (PREDICT/ABSTAIN at a threshold chosen by a harmonic-mean sweep over coverage and accuracy).

## Repository layout

```
configs/     hyperparameters: training, experiment matrix, local pipeline, EDOS pipeline
docs/        rules and invariants for the local pipeline (LOCAL_PIPELINE_RULES.md)
src/         shared library code
  data/      EXIST/EDOS loaders, preference-pair generation
  models/    baseline, zero/few-shot, SFT, DPO variants, agreement predictor
  pipeline/  6-stage experiment pipeline (openai, local, training, efficiency, reliability, report)
  utils/     metrics, calibration, reliability scoring, weight optimizer
scripts/
  run_pipeline.py       single entry point for the full experiment matrix
  analysis/             out-of-fold weight fitting, bootstrap statistics, subset analysis
  arr_ablations/        R(x) component ablation, threshold audit, matched-budget controls,
                        low-budget selection arms, EDOS second-dataset track
  local_pipeline/       hermetic 3B-backbone mirror of the hosted track (stages 01-07 + validator)
paper_assets/           shared figure style helpers
results/                per-instance prediction arrays (shipped); everything else is generated
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='sk-...'   # only needed for the hosted-model track
```

## Data (not redistributed here)

- **EXIST 2023**: request from the task organizers (http://nlp.uned.es/exist2023/). Place `EXIST2023_training.json` in the repository root and the official test/evaluation folder as `EXIST 2023 Dataset/`.
- **EDOS**: download from the authors' release (Kirk et al., 2023). Paths are set in `configs/edos_pipeline.yaml`.

## Running

```bash
# full pipeline (all stages, all models)
python scripts/run_pipeline.py

# specific stages
python scripts/run_pipeline.py --stages openai local
python scripts/run_pipeline.py --stages training efficiency
python scripts/run_pipeline.py --stages report

# small-sample debug run
python scripts/run_pipeline.py --max-samples 50

# local 3B-backbone mirror (fp16 on MPS, bitsandbytes on CUDA)
python scripts/local_pipeline/run_all.py
python scripts/local_pipeline/validate.py   # invariant checks; exit 0 = green

# EDOS track
python scripts/arr_ablations/edos_pipeline.py --config configs/edos_pipeline.yaml
```

`scripts/local_pipeline/validate.py` enforces the invariants in `docs/LOCAL_PIPELINE_RULES.md` (test-set size, bit-matching of shared arrays across tracks, weight normalization, config hashing) and fails closed.

## Notes

- Fine-tuned OpenAI model identifiers in `scripts/evaluate_all_models.py` have their organization segment replaced with `anonymized-org`; such models are only callable from the owning API organization in any case. Raw fine-tune job IDs (`ftjob-...`) carry no organization information and are likewise resolvable only inside the owning organization.
- `models/` and most of `results/` are produced by the pipeline. The per-instance prediction arrays behind the coverage-accuracy tables ship with the snapshot (`results/final_reliability_3factor/`, `results/local_pipeline/per_instance/`, `results/edos_pipeline/per_instance/`); they contain predictions, confidences, agreement scores, and token-uncertainty scores only, no dataset text.
