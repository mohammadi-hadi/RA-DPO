<div align="center">

# RA-DPO

### Reliability-Aware Preference Optimization and Selective Prediction for Sexism Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Datasets: EXIST 2023 · EDOS](https://img.shields.io/badge/Datasets-EXIST%202023%20%C2%B7%20EDOS-8A2BE2.svg)](#data)
[![Backbones: gpt-4o · Llama-3.2-3B · Qwen2.5-3B](https://img.shields.io/badge/Backbones-gpt--4o%20%C2%B7%20Llama--3.2--3B%20%C2%B7%20Qwen2.5--3B-0A7E8C.svg)](#results)
[![Languages: EN · ES](https://img.shields.io/badge/Languages-EN%20%C2%B7%20ES-success.svg)](#results)
[![Paper: under review](https://img.shields.io/badge/Paper-under%20review-lightgrey.svg)](#citation)

*Sexism detection is subjective: six annotators often disagree about the same post. RA-DPO
turns that disagreement into a reliability score that weights preference pairs during DPO
training and, more importantly, decides when the model should abstain instead of guess.*

</div>

---

## Overview

Most sexism-detection systems collapse multi-annotator labels into one gold label and treat
every example the same. That discards two signals about how trustworthy each prediction is:
how much the annotators agreed, and how confident the model is. RA-DPO
(Reliability-Aware Direct Preference Optimization) combines both into a single score

```
R(x) = alpha * confidence + beta * agreement + gamma * (1 - token_uncertainty)
```

whose weights are learned by out-of-fold logistic regression (5-fold, leakage-free). The
score has two jobs:

- **Training** — the model-independent part (agreement + token uncertainty) ranks the
  preference pairs; DPO trains on the most reliable subset, or on all pairs with
  reliability-weighted duplication (the RA-DPO variant).
- **Inference** — the full score gates abstention: the model answers only when
  `R(x) >= tau`, with `tau` chosen by a harmonic-mean sweep over accuracy and coverage.

Everything is evaluated on two corpora that publish per-annotator labels — EXIST 2023
(six annotators, EN/ES) and EDOS (three annotators, EN) — and on three backbones:
OpenAI `gpt-4o` plus two open-weight 3B models.

## Key Features

- **One score, two uses** — the same reliability estimate selects/weights training pairs and
  drives PREDICT/ABSTAIN at inference
- **Three agreement settings** — true annotator agreement (upper bound), agreement predicted
  from text by a fine-tuned encoder (deployable), and no agreement (floor)
- **Honest matched-budget controls** — every selection arm has a size-matched random
  counterpart, so "selection vs. less data" is answered by experiment, not assumption
- **Decision-threshold audit** — separates real fine-tuning gains from what a tuned decision
  threshold on the untrained base can already reach
- **Hermetic local mirror** — the 3B track re-implements the hosted track stage by stage,
  with a validator that bit-checks shared arrays and fails closed
- **Shipped per-instance arrays** — predictions, confidences, agreements, and
  token-uncertainty scores for every model (no dataset text), so the coverage-accuracy
  tables reproduce without any API access

<a name="results"></a>
## Results

Macro-F1 on the five corpus-backbone tracks (test sets: EXIST n=692, EDOS n=4,000):

| Variant | EXIST gpt-4o | EXIST Llama | EXIST Qwen | EDOS Llama | EDOS Qwen |
|---------|--------------|-------------|------------|------------|-----------|
| Base (prompted) | 0.724 | 0.522 | 0.524 | 0.626 | 0.666 |
| SFT | 0.820 | 0.369 | **0.580** | 0.681 | 0.546 |
| Standard DPO | 0.821 | 0.596 | 0.566 | 0.681 | 0.711 |
| Smart-30% | 0.821 | 0.574 | 0.530 | 0.656 | 0.698 |
| RA-DPO | **0.826** | **0.600** | 0.572 | **0.689** | **0.725** |

Selective prediction with `R(x)` on gpt-4o RA-DPO, accuracy at 50% coverage:

| Setting | Acc@50 |
|---------|--------|
| True agreement (upper bound) | 0.962 |
| Predicted agreement (deployable) | 0.887 |
| Confidence only (MaxProb baseline) | 0.879 |
| No agreement (floor) | 0.853 |

### Key findings

- **Abstention is where the score earns its keep.** The full `R(x)` exceeds the MaxProb
  confidence baseline by 8.4 points at 50% coverage on gpt-4o, and reliability-gated
  abstention improves every training variant on both corpora and all three backbones.
- **Selection does not beat random.** At a matched pair budget, reliability-ranked subsets
  and random subsets are statistically tied in all five controls (McNemar p = 0.22–1.00).
  The training-side result is data efficiency — 30% of the pairs matches full-data DPO with
  3.3x less data — not smarter pair picking.
- **The threshold audit separates learning from recalibration.** Every fine-tuned gpt-4o
  variant except the Ambiguous-only control exceeds the base model's best-threshold ceiling
  (0.800–0.826 vs. 0.764), so the hosted-track gains are real learning. Low-budget gains on
  the 3B backbones do not clear that bar, and both SFT collapses (0.369, 0.546) recover
  under threshold tuning — calibration failures, not ranking failures.
- **The learned weights track the annotation protocol.** With six annotators (EXIST),
  agreement carries most of the weight; with three (EDOS), the agreement signal is coarser
  and confidence takes over. `R(x)` recovers whichever signal is informative.
- **The deployable setting keeps most of the headroom.** Replacing true agreement with a
  text-based regressor (Twitter-XLM-R, Pearson r = 0.351) still exceeds both the
  confidence-only baseline and the no-agreement floor.

## Quick Start

```bash
git clone <this repository>
cd RA-DPO

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY='sk-...'   # only needed for the hosted-model track
```

Full pipeline (all stages, all models):

```bash
python scripts/run_pipeline.py                      # everything
python scripts/run_pipeline.py --stages openai local
python scripts/run_pipeline.py --stages training efficiency
python scripts/run_pipeline.py --max-samples 50     # small-sample debug run
```

Local 3B mirror (no API key; fp16 on Apple Silicon, bitsandbytes on CUDA):

```bash
python scripts/local_pipeline/run_all.py            # stages 01-07, resume-safe
python scripts/local_pipeline/validate.py           # invariant checks; exit 0 = green
```

EDOS second-dataset track:

```bash
python scripts/arr_ablations/edos_pipeline.py --config configs/edos_pipeline.yaml
```

The validator enforces the invariants in `docs/LOCAL_PIPELINE_RULES.md` (test-set sizes,
bit-matching of shared arrays across tracks, weight normalization, config hashing) and
fails closed: any violation exits non-zero.

## Why Abstain Instead of Select?

Both uses of `R(x)` were tested against controls, and only one survived. Ranking training
pairs by reliability never separates from a random draw of the same size — the top of the
ranking is dominated by unanimous examples, and narrowing the training distribution costs as
much as the cleaner labels gain. Ranking *test* examples by the same score is a different
story: examples the annotators disagreed on are exactly the ones the model gets wrong, so
routing them to abstention raises accuracy on everything the model still answers. The paper's
conclusion in one line: annotator disagreement is worth more as an abstention signal than as
a training filter.

## Repository Structure

```
RA-DPO/
├── configs/                 hyperparameters: training, experiment matrix, local + EDOS pipelines
├── docs/                    LOCAL_PIPELINE_RULES.md — invariants the validator enforces
├── src/
│   ├── data/                EXIST/EDOS loaders, preference-pair generation
│   ├── models/              baseline, zero/few-shot, SFT, DPO variants, agreement predictor
│   ├── pipeline/            6-stage experiment pipeline (openai, local, training, efficiency,
│   │                        reliability, report)
│   ├── explainability/      continuous token scoring (sigmoid token weights)
│   └── utils/               metrics, calibration, reliability scoring, weight optimizer
├── scripts/
│   ├── run_pipeline.py      single entry point for the full experiment matrix
│   ├── analysis/            OOF weight fitting, bootstrap statistics, subset analysis
│   ├── arr_ablations/       R(x) component ablation, threshold audit, matched-budget
│   │                        controls, low-budget selection arms, EDOS track
│   └── local_pipeline/      hermetic 3B mirror of the hosted track (stages 01-07 + validator)
├── paper_assets/            shared figure-style helpers
├── results/                 per-instance prediction arrays (shipped); everything else generated
└── requirements.txt
```

<a name="data"></a>
## Data

Neither dataset is redistributed here.

- **EXIST 2023** — request from the organizers via the
  [EXIST 2023 site](http://nlp.uned.es/exist2023/). Place `EXIST2023_training.json` in the
  repository root and the official test/evaluation folder as `EXIST 2023 Dataset/`.
- **EDOS** (SemEval-2023 Task 10) — download from the authors' release; paths are set in
  `configs/edos_pipeline.yaml`.

The shipped per-instance arrays in `results/` contain predictions, confidences, agreement
scores, and token-uncertainty scores only — no dataset text — so the coverage-accuracy
tables are reproducible without the raw data.

<a name="citation"></a>
## Citation

The accompanying paper is under review; citation information will be added upon publication.

## License

The code and result files are released under the MIT License (see [LICENSE](LICENSE)). The
EXIST 2023 and EDOS datasets are not included and are governed by their own usage terms.

## Notes

- Fine-tuned OpenAI model identifiers in `scripts/evaluate_all_models.py` have their
  organization segment replaced with `anonymized-org`; such models are only callable from
  the owning API organization in any case. Raw fine-tune job IDs (`ftjob-...`) carry no
  organization information and are likewise resolvable only inside the owning organization.
