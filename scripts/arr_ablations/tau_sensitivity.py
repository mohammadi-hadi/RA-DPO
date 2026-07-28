"""Abstention-threshold (tau) sensitivity analysis for the ARR revision.

Reviewer question: how does coverage/accuracy vary with the abstention
threshold tau on unseen data?  The paper selects tau* by maximizing the
harmonic mean of accuracy and coverage over a 50-point sweep.

For gpt-4o {base, SFT, Standard DPO, RA-DPO} and the two local RA-DPO
backbones (Llama-3.2-3B, Qwen2.5-3B) this script:

1. Tabulates coverage/accuracy vs tau from the stored ``sweep_results``
   (recomputing the sweep with the identical recipe when a file predates
   the sweep field: gpt-4o SFT and the local per-instance files).
2. Reports stability bands around the stored optimum tau*:
   min/max accuracy and coverage for tau in [tau*-0.05, tau*+0.05] and
   [tau*-0.10, tau*+0.10], plus the widest contiguous tau interval whose
   accuracy stays within 1pp of the accuracy at tau*.
3. Split-half generalization: 200 random half/half splits of the 692
   test items; tau is fit by harmonic-mean maximization on half A and
   evaluated on half B (mean +/- sd of B accuracy, coverage, and tau).
   This estimates deployment behavior when tau is tuned on one sample
   and applied to another.
4. Renders fig_tau_sensitivity.pdf/.png (accuracy and coverage vs tau
   for gpt-4o RA-DPO, tau* marked) in the paper figure style.

SELF-CHECK: at the stored tau* for gpt-4o RA-DPO the sweep row must
match the JSON ``reliability`` fields exactly, and R(x) recomputed from
``per_instance`` with the stored ``optimized_weights`` must reproduce
the stored sweep accuracy/coverage at three tau points within 0.01.

Outputs
-------
results/arr_ablations/tau_sensitivity/tau_curves.csv
results/arr_ablations/tau_sensitivity/splithalf.json
arr_revision/experiments/figures/fig_tau_sensitivity.{pdf,png}
arr_revision/experiments/tables/tab_tau_sensitivity.tex
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper_assets"))
from data import COLORS, apply_style  # noqa: E402  (paper figure style)

import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = ROOT / "results" / "arr_ablations" / "tau_sensitivity"
FIG_DIR = ROOT / "arr_revision" / "experiments" / "figures"
TAB_DIR = ROOT / "arr_revision" / "experiments" / "tables"

TAU_GRID = np.linspace(0.3, 0.95, 50)  # identical to evaluate_all_models.py
N_SPLITS = 200
SEED = 42
BAND_HALF_WIDTHS = (0.05, 0.10)
ACC_TOL_PP = 0.01  # 1pp tolerance for the stable interval

MODELS = [
    # (display name, json path)
    ("Base (gpt-4o)", ROOT / "results/final_reliability_3factor/gpt-4o_base.json"),
    ("SFT (gpt-4o)", ROOT / "results/final_reliability_3factor/gpt-4o_SFT.json"),
    ("Standard DPO (gpt-4o)", ROOT / "results/final_reliability_3factor/gpt-4o_Standard_DPO.json"),
    ("RA-DPO (gpt-4o)", ROOT / "results/final_reliability_3factor/gpt-4o_RA-DPO.json"),
    ("RA-DPO (Llama-3.2-3B)", ROOT / "results/local_pipeline/per_instance/llama32_3b_ra_dpo_local.json"),
    ("RA-DPO (Qwen2.5-3B)", ROOT / "results/local_pipeline/per_instance/qwen25_3b_ra_dpo_local.json"),
]

HEADLINE_MODEL = "RA-DPO (gpt-4o)"


# ---------------------------------------------------------------------------
# Core sweep machinery (mirrors scripts/evaluate_all_models.py exactly)
# ---------------------------------------------------------------------------

def compute_rx(per_instance: dict, weights: dict) -> tuple[np.ndarray, np.ndarray]:
    """R(x) = alpha*conf + beta*agree + gamma*(1 - sigmoid_score)."""
    conf = np.asarray(per_instance["confidences"], dtype=float)
    agree = np.asarray(per_instance["agreements"], dtype=float)
    sig = np.asarray(per_instance["sigmoid_scores"], dtype=float)
    correct = np.asarray(per_instance["correct"], dtype=bool)
    rx = (weights["alpha"] * conf
          + weights["beta"] * agree
          + weights["gamma"] * (1.0 - sig))
    return rx, correct


def sweep(rx: np.ndarray, correct: np.ndarray,
          taus: np.ndarray = TAU_GRID) -> list[dict]:
    """Coverage/accuracy/harmonic-mean at each tau (PREDICT iff R(x) >= tau)."""
    rows = []
    for tau in taus:
        mask = rx >= tau
        n_pred = int(mask.sum())
        if n_pred == 0:
            rows.append({"threshold": float(tau), "coverage": 0.0,
                         "accuracy": 0.0, "f1_combined": 0.0, "n_predicted": 0})
            continue
        cov = float(mask.mean())
        acc = float(correct[mask].mean())
        f1 = 2 * acc * cov / (acc + cov) if (acc + cov) > 0 else 0.0
        rows.append({"threshold": float(tau), "coverage": cov,
                     "accuracy": acc, "f1_combined": f1, "n_predicted": n_pred})
    return rows


def best_row(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: r["f1_combined"])


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def stability_bands(rows: list[dict], tau_star: float) -> dict:
    """Min/max accuracy and coverage in windows around tau*, plus the widest
    contiguous grid interval containing tau* with |acc - acc(tau*)| <= 1pp."""
    taus = np.array([r["threshold"] for r in rows])
    accs = np.array([r["accuracy"] for r in rows])
    covs = np.array([r["coverage"] for r in rows])
    i_star = int(np.argmin(np.abs(taus - tau_star)))
    acc_star = accs[i_star]

    out = {}
    for hw in BAND_HALF_WIDTHS:
        m = (taus >= tau_star - hw - 1e-12) & (taus <= tau_star + hw + 1e-12)
        out[f"band_pm{int(hw * 100):02d}"] = {
            "acc_min": float(accs[m].min()), "acc_max": float(accs[m].max()),
            "cov_min": float(covs[m].min()), "cov_max": float(covs[m].max()),
            "n_grid_points": int(m.sum()),
        }

    within = np.abs(accs - acc_star) <= ACC_TOL_PP + 1e-12
    lo, hi = i_star, i_star
    while lo > 0 and within[lo - 1]:
        lo -= 1
    while hi < len(taus) - 1 and within[hi + 1]:
        hi += 1
    out["stable_interval"] = {
        "tau_lo": float(taus[lo]), "tau_hi": float(taus[hi]),
        "width": float(taus[hi] - taus[lo]),
        "acc_at_tau_star": float(acc_star),
    }
    return out


def split_half(rx: np.ndarray, correct: np.ndarray,
               n_splits: int = N_SPLITS, seed: int = SEED) -> dict:
    """Fit tau on half A (harmonic-mean maximization), evaluate on half B."""
    rng = np.random.default_rng(seed)
    n = len(rx)
    taus_sel, accs_b, covs_b = [], [], []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        a, b = perm[: n // 2], perm[n // 2:]
        tau_a = best_row(sweep(rx[a], correct[a]))["threshold"]
        mask_b = rx[b] >= tau_a
        cov_b = float(mask_b.mean())
        acc_b = float(correct[b][mask_b].mean()) if mask_b.sum() else 0.0
        taus_sel.append(tau_a)
        accs_b.append(acc_b)
        covs_b.append(cov_b)
    return {
        "n_splits": n_splits, "seed": seed,
        "tau_mean": float(np.mean(taus_sel)), "tau_sd": float(np.std(taus_sel)),
        "acc_mean": float(np.mean(accs_b)), "acc_sd": float(np.std(accs_b)),
        "cov_mean": float(np.mean(covs_b)), "cov_sd": float(np.std(covs_b)),
    }


def verify_reconstruction(rows_stored: list[dict], rx: np.ndarray,
                          correct: np.ndarray, check_idx=(10, 25, 40)) -> dict:
    """Recomputed R(x) ranking must reproduce the stored sweep within 0.01."""
    checks = []
    ok = True
    for i in check_idx:
        s = rows_stored[i]
        mask = rx >= s["threshold"]
        acc = float(correct[mask].mean()) if mask.sum() else 0.0
        cov = float(mask.mean())
        d_acc = abs(acc - s["accuracy"])
        d_cov = abs(cov - s["coverage"])
        ok &= (d_acc <= 0.01) and (d_cov <= 0.01)
        checks.append({"threshold": s["threshold"],
                       "stored_acc": s["accuracy"], "recomputed_acc": acc,
                       "stored_cov": s["coverage"], "recomputed_cov": cov,
                       "abs_diff_acc": d_acc, "abs_diff_cov": d_cov})
    return {"pass": bool(ok), "points": checks}


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(rows: list[dict], tau_star: float, save_path: Path) -> None:
    apply_style()
    taus = [r["threshold"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    covs = [r["coverage"] for r in rows]

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(taus, accs, marker="o", color=COLORS["ra_dpo"], linewidth=1.8,
            markersize=4, label="Accuracy (on predicted)")
    ax.plot(taus, covs, marker="s", color=COLORS["standard"], linewidth=1.8,
            markersize=4, label="Coverage")
    ax.axvline(tau_star, color="#555555", linestyle="--", linewidth=1.2)
    ax.annotate(rf"$\tau^*={tau_star:.3f}$", xy=(tau_star, 0.42),
                xytext=(tau_star + 0.02, 0.40), fontsize=9, color="#555555")
    ax.set_xlabel(r"Abstention threshold $\tau$")
    ax.set_ylabel("Accuracy / Coverage")
    ax.set_title("RA-DPO (gpt-4o)")
    ax.set_xlim(0.28, 0.97)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    fig.savefig(save_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"saved -> {save_path}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def make_table(results: dict, save_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        "",
        r"\caption{Sensitivity of the abstention threshold $\tau$."
        r" $\tau^*$ maximizes the harmonic mean of accuracy and coverage on"
        r" the test sweep; the accuracy range is over"
        r" $\tau \in [\tau^*\!-\!0.05, \tau^*\!+\!0.05]$; the stable"
        r" interval is the widest contiguous $\tau$ range whose accuracy"
        r" stays within 1pp of accuracy at $\tau^*$. Split-half: $\tau$"
        r" fit on one random half of the test set and evaluated on the"
        r" other (200 splits, mean $\pm$ sd).}",
        r"\label{tab:tau_sensitivity}",
        "",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & $\tau^*$ & Acc@$\tau^*$ & Cov@$\tau^*$ &"
        r" Acc range $\pm 0.05$ & Stable width & Split-half acc \\",
        r"\midrule",
    ]
    for name, res in results.items():
        b = res["bands"]["band_pm05"]
        si = res["bands"]["stable_interval"]
        sh = res["splithalf"]
        lines.append(
            f"{name} & {res['tau_star']:.3f} & {res['acc_at_tau_star']:.3f}"
            f" & {res['cov_at_tau_star']:.3f}"
            f" & {b['acc_min']:.3f}--{b['acc_max']:.3f}"
            f" & {si['width']:.3f}"
            f" & {sh['acc_mean']:.3f} $\\pm$ {sh['acc_sd']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text("\n".join(lines))
    print(f"saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    csv_rows: list[dict] = []
    selfcheck_pass = True
    selfcheck_msgs: list[str] = []

    for name, path in MODELS:
        d = json.loads(path.read_text())
        rx, correct = compute_rx(d["per_instance"], d["optimized_weights"])

        stored = d.get("sweep_results")
        if stored is not None:
            rows = stored
            source = "stored"
            ver = verify_reconstruction(stored, rx, correct)
            if not ver["pass"]:
                selfcheck_pass = False
                selfcheck_msgs.append(f"[FAIL] {name}: R(x) reconstruction "
                                      f"deviates > 0.01 from stored sweep")
        else:
            rows = sweep(rx, correct)
            source = "recomputed"
            ver = {"pass": True, "points": [],
                   "note": "no stored sweep; recomputed with identical recipe"}

        rel = d.get("reliability")
        if rel is not None:
            tau_star = rel["optimal_threshold"]
            acc_star, cov_star = rel["accuracy_at_optimal"], rel["coverage_at_optimal"]
        else:
            br = best_row(rows)
            tau_star, acc_star, cov_star = br["threshold"], br["accuracy"], br["coverage"]

        # Consistency: sweep row at tau* must agree with the reliability fields.
        row_star = min(rows, key=lambda r: abs(r["threshold"] - tau_star))
        d_acc = abs(row_star["accuracy"] - acc_star)
        d_cov = abs(row_star["coverage"] - cov_star)
        if name == HEADLINE_MODEL:
            ok = (d_acc < 1e-9) and (d_cov < 1e-9) and ver["pass"]
            selfcheck_pass &= ok
            selfcheck_msgs.append(
                f"[{'PASS' if ok else 'FAIL'}] {name}: reliability fields "
                f"tau*={tau_star:.4f} acc={acc_star:.4f} cov={cov_star:.4f} vs "
                f"sweep row acc={row_star['accuracy']:.4f} "
                f"cov={row_star['coverage']:.4f} "
                f"(|d_acc|={d_acc:.2e}, |d_cov|={d_cov:.2e}); "
                f"R(x) reconstruction max |d_acc| over checked points = "
                f"{max((p['abs_diff_acc'] for p in ver['points']), default=0.0):.2e}"
            )
            # acc@100% must equal sweep accuracy at full coverage
            full = rows[0]
            acc100 = d["accuracy_at_coverage"]["acc@100%"]
            ok100 = full["coverage"] == 1.0 and abs(full["accuracy"] - acc100) < 1e-9
            selfcheck_pass &= ok100
            selfcheck_msgs.append(
                f"[{'PASS' if ok100 else 'FAIL'}] {name}: sweep acc at full "
                f"coverage {full['accuracy']:.4f} == acc@100% {acc100:.4f}"
            )

        bands = stability_bands(rows, tau_star)
        sh = split_half(rx, correct)

        all_results[name] = {
            "source": source,
            "tau_star": float(tau_star),
            "acc_at_tau_star": float(acc_star),
            "cov_at_tau_star": float(cov_star),
            "bands": bands,
            "splithalf": sh,
            "verification": ver,
        }
        for r in rows:
            csv_rows.append({"model": name, "source": source, **{
                k: r[k] for k in
                ("threshold", "coverage", "accuracy", "f1_combined", "n_predicted")}})

        if name == HEADLINE_MODEL:
            make_figure(rows, tau_star, FIG_DIR / "fig_tau_sensitivity.pdf")

    # ---- outputs ----------------------------------------------------------
    csv_path = OUT_DIR / "tau_curves.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"saved -> {csv_path}")

    sh_path = OUT_DIR / "splithalf.json"
    sh_path.write_text(json.dumps(all_results, indent=2))
    print(f"saved -> {sh_path}")

    make_table(all_results, TAB_DIR / "tab_tau_sensitivity.tex")

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 118)
    print(f"{'Model':<24} {'sweep':>10} {'tau*':>7} {'acc@t*':>7} {'cov@t*':>7} "
          f"{'acc range +/-0.05':>18} {'stable width':>13} "
          f"{'splithalf acc':>15} {'splithalf cov':>15} {'tau (A)':>13}")
    print("-" * 118)
    for name, res in all_results.items():
        b = res["bands"]["band_pm05"]
        si = res["bands"]["stable_interval"]
        sh = res["splithalf"]
        print(f"{name:<24} {res['source']:>10} {res['tau_star']:>7.3f} "
              f"{res['acc_at_tau_star']:>7.3f} {res['cov_at_tau_star']:>7.3f} "
              f"{b['acc_min']:>8.3f}--{b['acc_max']:<8.3f} {si['width']:>13.3f} "
              f"{sh['acc_mean']:>7.3f}+/-{sh['acc_sd']:<5.3f} "
              f"{sh['cov_mean']:>7.3f}+/-{sh['cov_sd']:<5.3f} "
              f"{sh['tau_mean']:>6.3f}+/-{sh['tau_sd']:<5.3f}")
    print("=" * 118)

    print("\nSELF-CHECK:")
    for m in selfcheck_msgs:
        print(" ", m)
    print(f"SELF-CHECK {'PASSED' if selfcheck_pass else 'FAILED'}")
    return 0 if selfcheck_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
