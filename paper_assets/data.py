"""Single source of truth for every number in the paper.

Each asset script (fig_*.py, tab_*.py) imports the relevant block from here.
The Jupyter notebook also hard-codes these same numbers inline so it is
fully standalone.

Update here if numbers change; rerun the asset scripts or the notebook.
"""
from __future__ import annotations
import matplotlib as _mpl
import matplotlib.pyplot as _plt

# ---- Global plot style ----------------------------------------------------
PLOT_STYLE = {
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_style() -> None:
    _mpl.rcParams.update(PLOT_STYLE)


# Brand colors
COLORS = {
    "base": "#8A8A8A",
    "sft": "#E8763B",
    "smart10": "#9AC0CD",
    "smart30": "#F5A623",
    "smart50": "#F5D76E",
    "random50": "#B8A9C9",
    "standard": "#4A90E2",
    "ambiguous": "#C0392B",
    "ra_dpo": "#D62828",
    "fine_tuned": "#27AE60",
}

# ---- Paper numbers --------------------------------------------------------
# Reference: unified_gpt4o / final_reliability_3factor, 2026-04-15 run.
# Test set: 692 EN+ES samples, structured prompt, 5-fold OOF R(x) weights.

# Table 1: optimized R(x) weights (OOF mean)
WEIGHTS = {
    # label → (alpha, beta, gamma)
    "OpenAI (base)":            (0.378, 0.536, 0.086),
    "OpenAI (SFT)":             (0.280, 0.682, 0.038),
    "OpenAI (Smart-10% DPO)":   (0.055, 0.897, 0.048),
    "OpenAI (Smart-30% DPO)":   (0.084, 0.843, 0.073),
    "OpenAI (Smart-50% DPO)":   (0.113, 0.836, 0.052),
    "OpenAI (Random-50% DPO)":  (0.129, 0.791, 0.079),
    "OpenAI (Ambiguous-only)":  (0.171, 0.621, 0.207),
    "OpenAI (Standard DPO)":    (0.059, 0.809, 0.133),
    "OpenAI (RA-DPO)":          (0.146, 0.827, 0.027),
    "shared (cross-validated)": (0.131, 0.793, 0.076),
    "RA-DPO (predicted-agreement)": (0.260, 0.690, 0.050),
    # --- Local pipeline R(x) weights (per-variant, true-agreement,
    # source: results/local_pipeline/unified/{qwen25_3b,llama32_3b}/weights.csv) ---
    "Qwen (SFT)":                (0.082, 0.608, 0.310),
    "Qwen (Smart-10% DPO)":      (0.230, 0.500, 0.269),
    "Qwen (Smart-30% DPO)":      (0.216, 0.515, 0.269),
    "Qwen (Smart-50% DPO)":      (0.164, 0.591, 0.245),
    "Qwen (Random-50% DPO)":     (0.142, 0.608, 0.250),
    "Qwen (Standard DPO)":       (0.158, 0.591, 0.251),
    "Qwen (Ambiguous-only DPO)": (0.275, 0.487, 0.239),
    "Qwen (RA-DPO)":             (0.147, 0.624, 0.229),
    "Llama (SFT)":                (0.637, 0.164, 0.199),
    "Llama (Smart-10% DPO)":      (0.358, 0.335, 0.308),
    "Llama (Smart-30% DPO)":      (0.067, 0.535, 0.398),
    "Llama (Smart-50% DPO)":      (0.066, 0.673, 0.261),
    "Llama (Random-50% DPO)":     (0.042, 0.646, 0.312),
    "Llama (Standard DPO)":       (0.240, 0.572, 0.189),
    "Llama (Ambiguous-only DPO)": (0.086, 0.440, 0.475),
    "Llama (RA-DPO)":             (0.193, 0.668, 0.140),
}

# Table 2: best F1 per model (prompt study, all strategies)
MODEL_BEST = [
    ("gpt-4o-mini (SFT fine-tuned)", 0.842, "definition/ZS"),
    ("gpt-4o",                       0.803, "persona/FS"),
    ("gpt-4o-mini",                  0.792, "structured/FS"),
    ("gpt-4.1-mini",                 0.777, "persona/FS"),
    ("gpt-4.1",                      0.772, "persona/FS"),
    ("gpt-5.4-mini",                 0.768, "definition/FS"),
    ("gpt-4.1-nano",                 0.762, "definition/ZS"),
    ("gpt-5.4-nano",                 0.749, "structured/ZS"),
    ("gpt-5.4",                      0.746, "structured/FS"),
    ("o3 (reasoning)",               0.573, "cot/FS"),
]

# Table 3: avg F1 by prompt strategy (across 9 LLMs, excl. o3)
PROMPT_STRATEGY = [
    ("Structured",      0.741, 0.041),
    ("Definition",      0.738, 0.039),
    ("Persona",         0.734, 0.050),
    ("Basic",           0.673, 0.084),
    ("Chain-of-Thought",0.665, 0.090),
]

# Table 4: fine-tuning results (single OpenAI base model, structured prompt)
FINETUNING = [
    # label, pairs, F1, CI low, CI high
    ("OpenAI (base)",                 None,  0.723, 0.689, 0.757),
    ("OpenAI (Ambiguous-only DPO)",   665,   0.697, None,  None),
    ("OpenAI (Smart-10% DPO)",        554,   0.802, 0.770, 0.834),
    ("OpenAI (Random-50% DPO)",       2768,  0.814, 0.783, 0.842),
    ("OpenAI (Smart-50% DPO)",        2768,  0.819, 0.790, 0.847),
    ("OpenAI (SFT)",                  5535,  0.820, 0.791, 0.849),
    ("OpenAI (Standard DPO)",         5536,  0.821, 0.793, 0.850),
    ("OpenAI (Smart-30% DPO)",        1661,  0.821, 0.792, 0.850),
    ("OpenAI (RA-DPO)",               8984,  0.826, 0.797, 0.855),
]

# Table 6: coverage-accuracy under true-agreement / predicted-agreement / no-agreement
# {(model, setting): {coverage: acc}}
COVERAGE = {
    ("OpenAI (base)",            "true-agreement"):       {100: 0.740, 90: 0.766, 80: 0.798, 60: 0.834, 50: 0.879},
    ("OpenAI (base)",            "predicted-agreement"):  {100: 0.740, 60: 0.800, 50: 0.824},
    ("OpenAI (base)",            "no-agreement"):         {100: 0.740, 60: 0.793, 50: 0.775},
    ("OpenAI (SFT)",             "true-agreement"):       {100: 0.820, 90: 0.860, 80: 0.906, 60: 0.942, 50: 0.965},
    ("OpenAI (Smart-10% DPO)",   "true-agreement"):       {100: 0.802, 90: 0.835, 80: 0.866, 60: 0.911, 50: 0.925},
    ("OpenAI (Smart-30% DPO)",   "true-agreement"):       {100: 0.824, 90: 0.864, 80: 0.894, 60: 0.937, 50: 0.945},
    ("OpenAI (Smart-30% DPO)",   "predicted-agreement"):  {100: 0.824, 60: 0.851, 50: 0.858},
    ("OpenAI (Smart-50% DPO)",   "true-agreement"):       {100: 0.821, 90: 0.865, 80: 0.892, 60: 0.937, 50: 0.939},
    ("OpenAI (Random-50% DPO)",  "true-agreement"):       {100: 0.817, 90: 0.849, 80: 0.872, 60: 0.935, 50: 0.945},
    ("OpenAI (Standard DPO)",    "true-agreement"):       {100: 0.825, 90: 0.857, 80: 0.879, 60: 0.925, 50: 0.922},
    ("OpenAI (Standard DPO)",    "predicted-agreement"):  {100: 0.825, 60: 0.863, 50: 0.870},
    ("OpenAI (Ambiguous-only DPO)", "true-agreement"):    {100: 0.697, 90: 0.714, 80: 0.729, 60: 0.764, 50: 0.775},
    ("OpenAI (RA-DPO)",          "true-agreement"):       {100: 0.828, 90: 0.864, 80: 0.888, 60: 0.942, 50: 0.962},
    ("OpenAI (RA-DPO)",          "predicted-agreement"):  {100: 0.828, 60: 0.870, 50: 0.884},
    ("OpenAI (RA-DPO)",          "no-agreement"):         {100: 0.828, 60: 0.848, 50: 0.853},
    # --- Local models (true-agreement only, from results/local_pipeline/unified/*/coverage_accuracy.csv) ---
    ("Qwen (SFT)",                  "true-agreement"):    {100: 0.6301, 90: 0.6388, 80: 0.6462, 60: 0.6747, 50: 0.6792},
    ("Qwen (Smart-10% DPO)",        "true-agreement"):    {100: 0.6098, 90: 0.6083, 80: 0.6191, 60: 0.6554, 50: 0.6850},
    ("Qwen (Smart-30% DPO)",        "true-agreement"):    {100: 0.6142, 90: 0.6148, 80: 0.6137, 60: 0.6627, 50: 0.6792},
    ("Qwen (Smart-50% DPO)",        "true-agreement"):    {100: 0.6228, 90: 0.6324, 80: 0.6372, 60: 0.6530, 50: 0.6618},
    ("Qwen (Random-50% DPO)",       "true-agreement"):    {100: 0.6228, 90: 0.6276, 80: 0.6372, 60: 0.6530, 50: 0.6647},
    ("Qwen (Standard DPO)",         "true-agreement"):    {100: 0.6315, 90: 0.6372, 80: 0.6534, 60: 0.6819, 50: 0.6908},
    ("Qwen (Ambiguous-only DPO)",   "true-agreement"):    {100: 0.6113, 90: 0.6148, 80: 0.6245, 60: 0.6482, 50: 0.6647},
    ("Qwen (RA-DPO)",               "true-agreement"):    {100: 0.6344, 90: 0.6388, 80: 0.6534, 60: 0.6867, 50: 0.6965},
    ("Llama (SFT)",                 "true-agreement"):    {100: 0.5564, 90: 0.5843, 80: 0.5866, 60: 0.6265, 50: 0.6329},
    ("Llama (Smart-10% DPO)",       "true-agreement"):    {100: 0.6026, 90: 0.6148, 80: 0.6191, 60: 0.6578, 50: 0.6561},
    ("Llama (Smart-30% DPO)",       "true-agreement"):    {100: 0.6315, 90: 0.6437, 80: 0.6552, 60: 0.6723, 50: 0.6676},
    ("Llama (Smart-50% DPO)",       "true-agreement"):    {100: 0.6358, 90: 0.6453, 80: 0.6625, 60: 0.6940, 50: 0.7110},
    ("Llama (Random-50% DPO)",      "true-agreement"):    {100: 0.6358, 90: 0.6517, 80: 0.6552, 60: 0.6916, 50: 0.7023},
    ("Llama (Standard DPO)",        "true-agreement"):    {100: 0.6315, 90: 0.6437, 80: 0.6606, 60: 0.6916, 50: 0.6965},
    ("Llama (Ambiguous-only DPO)",  "true-agreement"):    {100: 0.6171, 90: 0.6244, 80: 0.6209, 60: 0.6530, 50: 0.6647},
    ("Llama (RA-DPO)",              "true-agreement"):    {100: 0.6286, 90: 0.6453, 80: 0.6480, 60: 0.6771, 50: 0.6763},
}

# Table: training data efficiency
EFFICIENCY = [
    ("Smart-10% DPO",   554,  0.802),
    ("Smart-30% DPO",   1661, 0.821),
    ("Smart-50% DPO",   2768, 0.819),
    ("Random-50% DPO",  2768, 0.814),
    ("Standard DPO",    5536, 0.821),
    ("RA-DPO (oversampled)", 8984, 0.826),
]

# Table 5: subset composition (training-set breakdown)
SUBSET_COMPOSITION = [
    # subset, pairs, 3/3, 4/2, 5/1, 6/0 percentages
    ("Smart-10%",       554,  0,   0,   0,   100),
    ("Smart-30%",       1661, 0,   0,   0,   100),
    ("Smart-50%",       2768, 0,   0,   35,  65),
    ("Random-50%",      2768, 12,  26,  30,  32),
    ("Standard (100%)", 5536, 12,  26,  30,  33),
    ("Ambiguous-only",  665,  100, 0,   0,   0),
]

# Hard vs easy cases (test split by annotator agreement).
# Easy = 426 test instances with agreement >= 0.83 (5/6 or 6/6 majority).
# Hard = 266 test instances with agreement <  0.67 (3/3 and 4/2 splits).
# Values are the actual `per_instance.correct` rate from
# results/final_reliability_3factor/gpt-4o_*.json computed on the canonical
# 692-sample test split (random_state=42 stratified split, see
# ra_dpo/data/data_loader.py). One base model (gpt-4o), no gpt-4o-mini entries.
HARD_EASY = [
    # model, easy acc (>=0.83), hard acc (<0.67)
    ("OpenAI base",          0.819, 0.613),
    ("OpenAI SFT",           0.930, 0.650),
    ("OpenAI Std DPO",       0.925, 0.665),
    ("OpenAI Smart-30%",     0.937, 0.643),
    ("OpenAI RA-DPO",        0.939, 0.650),
]

# Language comparison (best model × lang)
LANGUAGE = [
    # model, EN F1, ES F1
    ("gpt-4o-mini (SFT)",  0.862, 0.822),
    ("gpt-4o",             0.821, 0.785),
    ("gpt-4o-mini",        0.789, 0.796),
    ("gpt-4.1-mini",       0.795, 0.758),
    ("gpt-4.1",            0.792, 0.750),
    ("gpt-5.4-mini",       0.785, 0.751),
    ("gpt-4.1-nano",       0.781, 0.743),
    ("gpt-5.4-nano",       0.770, 0.729),
    ("gpt-5.4",            0.764, 0.728),
    ("o3 (reasoning)",     0.585, 0.561),
]

# Prompt heatmap: F1 per (model, prompt_strategy)
PROMPT_HEATMAP_MODELS = [
    "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
    "gpt-4o-mini",  "gpt-4o",
    "gpt-5-nano",   "gpt-5-mini",   "gpt-5",
    "o3 (reasoning)",
]
PROMPT_HEATMAP_STRATS = ["basic", "cot", "persona", "definition", "structured"]
PROMPT_HEATMAP_VALUES = [
    # rows = models, columns = strategies (avg over ZS/FS, EN/ES)
    [0.708, 0.642, 0.710, 0.762, 0.755],  # gpt-4.1-nano
    [0.709, 0.688, 0.777, 0.750, 0.755],  # gpt-4.1-mini
    [0.691, 0.668, 0.772, 0.748, 0.761],  # gpt-4.1
    [0.735, 0.648, 0.761, 0.778, 0.792],  # gpt-4o-mini
    [0.700, 0.687, 0.803, 0.756, 0.763],  # gpt-4o
    [0.718, 0.641, 0.706, 0.748, 0.749],  # gpt-5-nano
    [0.708, 0.689, 0.755, 0.768, 0.761],  # gpt-5-mini
    [0.716, 0.671, 0.736, 0.734, 0.746],  # gpt-5
    [0.457, 0.573, 0.475, 0.498, 0.519],  # o3
]

# Local-model corroboration: F1, acc@100%, acc@50% for both 3B local models
# plus gpt-4o reference. Values from results/local_pipeline/unified/<model>/summary.json
# and coverage_accuracy.csv (true-agreement setting).
LOCAL_CORROBORATION = {
    # variant: {"openai": (F1, acc50), "qwen": (F1, acc50), "llama": (F1, acc50)}
    "base":         {"openai": (0.723, 0.879), "qwen": (0.524, None),  "llama": (0.522, None)},
    "sft":          {"openai": (0.820, 0.965), "qwen": (0.580, 0.679), "llama": (0.369, 0.633)},
    "smart10_dpo":  {"openai": (0.802, 0.925), "qwen": (0.521, 0.685), "llama": (0.510, 0.656)},
    "smart30_dpo":  {"openai": (0.821, 0.945), "qwen": (0.530, 0.679), "llama": (0.574, 0.668)},
    "smart50_dpo":  {"openai": (0.818, 0.939), "qwen": (0.550, 0.662), "llama": (0.594, 0.711)},
    "random50_dpo": {"openai": (0.813, 0.945), "qwen": (0.550, 0.665), "llama": (0.589, 0.702)},
    "std_dpo":      {"openai": (0.821, 0.922), "qwen": (0.566, 0.691), "llama": (0.596, 0.697)},
    "ambiguous_dpo":{"openai": (0.697, 0.775), "qwen": (0.525, 0.665), "llama": (0.537, 0.665)},
    "ra_dpo":       {"openai": (0.826, 0.962), "qwen": (0.572, 0.697), "llama": (0.600, 0.676)},
}

# Local DPO training losses (sanity check: ordering by training-set quality)
LOCAL_TRAIN_LOSS = {
    # variant: {"qwen": loss, "llama": loss}
    "ra_dpo":       {"qwen": 0.499, "llama": 0.499},
    "smart50_dpo":  {"qwen": 0.514, "llama": 0.522},
    "smart30_dpo":  {"qwen": 0.545, "llama": 0.547},
    "std_dpo":      {"qwen": 0.563, "llama": 0.572},
    "random50_dpo": {"qwen": 0.599, "llama": 0.607},
    "smart10_dpo":  {"qwen": 0.638, "llama": 0.647},
    "ambiguous_dpo":{"qwen": 0.693, "llama": 0.693},
}

# Training-progression story (for a single bar chart)
TRAINING_PROGRESSION = [
    # label, F1, track ("base", "sft", "dpo", "smart", "ra")
    ("OpenAI base",            0.724, "base"),
    ("OpenAI-mini (base)",     0.778, "base"),
    ("OpenAI-mini (SFT)",      0.825, "sft"),
    ("OpenAI SFT",             0.820, "sft"),
    ("OpenAI Random-50% DPO",  0.814, "dpo"),
    ("OpenAI Smart-50% DPO",   0.819, "smart"),
    ("OpenAI Smart-30% DPO",   0.821, "smart"),
    ("OpenAI Standard DPO",    0.821, "dpo"),
    ("OpenAI RA-DPO",          0.826, "ra"),
]
