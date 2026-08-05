"""RA-DPO: Reliability-Aware Preference Optimization and Selective Prediction.

Turns annotator disagreement into a reliability score
``R(x) = alpha * confidence + beta * agreement + gamma * (1 - token_uncertainty)``
that weights preference pairs during DPO training and gates abstention at
inference.

Subpackages:
    data            EXIST/EDOS loaders, preference-pair generation
    pipeline        experiment pipeline: config, prompts, runners, orchestrator
    utils           metrics, calibration, reliability scoring, weight optimizer
    explainability  token-level scoring and SHAP analysis (needs the ``explain`` extra)
"""

__version__ = "0.1.0"
