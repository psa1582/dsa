from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEST_PROMPTS = {
    "code_heldout_3",
    "code_heldout_5",
    "text_heldout_26493",
    "text_heldout_28444",
}


def aggregate_selection(frame: pd.DataFrame, scope: str, method: str) -> dict[str, float]:
    values = frame[(frame.scope == scope) & (frame.method == method) & (frame.step <= 64)]
    if values.empty:
        raise RuntimeError(f"missing {scope}/{method}")
    return {
        "observations": int(len(values)),
        "qk_reduction": float(values.qk_reduction.median()),
        "top2048_recall": float(values.top2048_recall.mean()),
        "top128_recall": float(values.top128_recall.mean()),
        "top512_recall": float(values.top512_recall.mean()),
        "top1024_recall": float(values.top1024_recall.mean()),
        "weighted_recall": float(values.weighted_recall.mean()),
        "teacher_attention_mass_ratio": float(values.teacher_attention_mass_ratio.mean()),
        "exact_top2048_match_rate": float(values.exact_top2048_match.mean()),
    }


def global_rescue(frame: pd.DataFrame) -> dict[str, float]:
    values = frame[
        frame.prompt_id.isin(TEST_PROMPTS)
        & frame.policy_role.eq("Aggressive")
        & frame.verifier.eq("head_dynamic_abs_w8_b64_r0.1")
    ]
    if len(values) != 6144:
        raise RuntimeError(f"expected 6144 global rescue rows, got {len(values)}")
    return {
        "observations": int(len(values)),
        "qk_reduction": float(values.net_qk_reduction.median()),
        "top2048_recall": float(values.top2048_recall.mean()),
        "top128_recall": float(values.top128_recall.mean()),
        "top512_recall": float(values.top512_recall.mean()),
        "top1024_recall": float(values.top1024_recall.mean()),
        "weighted_recall": float(values.index_mass_ratio.mean()),
        "teacher_attention_mass_ratio": float("nan"),
        "exact_top2048_match_rate": float(values.exact_match.mean()),
    }


def main_results(
    offline_rows: pd.DataFrame,
    rescue_rows: pd.DataFrame,
    mla: pd.Series,
    teacher: pd.Series,
) -> pd.DataFrame:
    definitions = [
        ("Full64 oracle", "reference", "full64_oracle", "All tokens", "Yes"),
        ("Existing H8 + 10% H56 rescue", "global", "global", "Temporal-hot + rescued cold", "Promoted cold blocks"),
        ("Raw H8", "pure_no_h56", "raw_h8", "None", "No"),
        ("Previous Full64 only", "pure_no_h56", "prev_full", "None", "No"),
        ("Per-layer affine H8", "pure_no_h56", "layer_affine_h8", "None", "No"),
        ("Per-layer T1 H8 + previous Full64", "pure_no_h56", "t1_layer", "None", "No"),
        ("Hybrid raw H8", "hybrid_temporal_hot_full_cold_h8", "raw_h8", "Temporal-hot", "Temporal-hot only"),
        ("Hybrid affine H8", "hybrid_temporal_hot_full_cold_h8", "layer_affine_h8", "Temporal-hot", "Temporal-hot only"),
        ("Hybrid T1 H8 + previous Full64", "hybrid_temporal_hot_full_cold_h8", "t1_layer", "Temporal-hot", "Temporal-hot only"),
    ]
    rows: list[dict[str, Any]] = []
    for label, scope, method, full_region, h56 in definitions:
        if scope == "global":
            metric = global_rescue(rescue_rows)
        else:
            metric = aggregate_selection(offline_rows, scope, method)
        is_rescue = scope == "global"
        is_full = label == "Full64 oracle"
        rows.append(
            {
                "method": label,
                "h56_computed": h56,
                "current_full64_region": full_region,
                "qk_reduction": metric["qk_reduction"],
                "top2048_recall": metric["top2048_recall"],
                "top128_recall": metric["top128_recall"],
                "top512_recall": metric["top512_recall"],
                "top1024_recall": metric["top1024_recall"],
                "weighted_recall": metric["weighted_recall"],
                "exact_top2048_match_rate": metric["exact_top2048_match_rate"],
                "teacher_attention_mass_ratio": metric["teacher_attention_mass_ratio"],
                "mla_relative_l2_p95": 0.0 if is_full else (float(mla.output_relative_l2_p95) if is_rescue else float("nan")),
                "mla_relative_l2_p99": 0.0 if is_full else (float(mla.output_relative_l2_p99) if is_rescue else float("nan")),
                "mla_cosine_p5": 1.0 if is_full else (float(mla.output_cosine_p5) if is_rescue else float("nan")),
                "logit_kl_mean": 0.0 if is_full else (float(teacher.logit_kl_mean) if is_rescue else float("nan")),
                "logit_kl_p95": 0.0 if is_full else (float(teacher.logit_kl_p95) if is_rescue else float("nan")),
                "ppl_delta": 0.0 if is_full else (float(teacher.ppl_delta) if is_rescue else float("nan")),
                "quality_status": "measured prior GPU replay" if is_rescue else ("oracle" if is_full else "GPU FOLLOW-UP REQUIRED"),
                "comparison_steps": 64,
                "test_trace_count": 96,
            }
        )
    return pd.DataFrame(rows)


def calibration_table(coefficients: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for layer in [2, 5, 8, 11, 14, 17, 21, 25]:
        a, b, c = coefficients[f"t1_layer_{layer}"]
        affine_a, affine_c = coefficients[f"affine_layer_{layer}"]
        rows.append(
            {
                "layer": layer,
                "t1_a_h8": a,
                "t1_b_previous_full": b,
                "t1_c_intercept": c,
                "affine_a_h8": affine_a,
                "affine_c_intercept": affine_c,
                "margin_band": coefficients["margin_band"][str(layer)],
                "calibration_observations": coefficients["sample_counts"][f"t1_layer_{layer}"],
            }
        )
    return pd.DataFrame(rows)


def cost_table() -> pd.DataFrame:
    rows = []
    for length in [16384, 32768, 65536, 131072]:
        indexer_k = length * 128 * 2
        for dtype, width in [("FP32", 4), ("FP16/BF16", 2), ("INT16", 2)]:
            per_layer = length * width
            rows.append(
                {
                    "sequence_length": length,
                    "scalar_dtype": dtype,
                    "bytes_per_scalar": width,
                    "previous_score_bytes_per_layer": per_layer,
                    "previous_score_bytes_8_layers": per_layer * 8,
                    "bf16_indexer_k_bytes_per_layer": indexer_k,
                    "scalar_vs_indexer_k_fraction": per_layer / indexer_k,
                    "full64_qk_macs_per_token": 64 * 128,
                    "h8_qk_macs_per_token": 8 * 128,
                    "pure_no_h56_qk_reduction": 0.875,
                    "t1_scalar_flops_per_token": 5,
                    "physical_k_bytes_read_per_token": 128 * 2,
                    "physical_k_byte_reduction_claimed": 0.0,
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, root: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(root / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def figures(
    samples_path: Path,
    main: pd.DataFrame,
    coefficients: dict[str, Any],
    rescue_mla_rows: pd.DataFrame,
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with np.load(samples_path) as payload:
        sample = {key: payload[key] for key in payload.files}
    count = sample["h8"].size
    ids = np.linspace(0, count - 1, min(60000, count), dtype=np.int64)
    newly = sample["newly_active"][ids].astype(bool)

    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.hexbin(sample["h8"][ids][~newly], sample["full"][ids][~newly], gridsize=80, bins="log", cmap="Blues", mincnt=1)
    axis.scatter(sample["h8"][ids][newly], sample["full"][ids][newly], s=5, c="#DC2626", alpha=0.35, label="newly-active")
    axis.set(title="Figure A — Current H8 vs Full64", xlabel="H8 score", ylabel="Full64 score")
    axis.legend(loc="best")
    save_figure(fig, output, "figure_a_h8_vs_full64.png")

    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.hexbin(sample["previous"][ids][~newly], sample["full"][ids][~newly], gridsize=80, bins="log", cmap="Greens", mincnt=1)
    axis.scatter(sample["previous"][ids][newly], sample["full"][ids][newly], s=5, c="#DC2626", alpha=0.35, label="newly-active")
    axis.set(title="Figure B — Previous Full64 vs current Full64", xlabel="Previous Full64", ylabel="Current Full64")
    axis.legend(loc="best")
    save_figure(fig, output, "figure_b_previous_vs_current_full64.png")

    a, b, c = coefficients["t2_global"]
    predicted_residual = a * sample["h8"][ids] + b * sample["previous"][ids] + c
    true_residual = sample["residual"][ids]
    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.hexbin(predicted_residual, true_residual, gridsize=80, bins="log", cmap="Purples", mincnt=1)
    lo = float(min(predicted_residual.min(), true_residual.min()))
    hi = float(max(predicted_residual.max(), true_residual.max()))
    axis.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1, label="ideal")
    axis.set(title="Figure C — True vs predicted H56 residual (global T2 diagnostic)", xlabel="Predicted H56 residual", ylabel="True H56 residual")
    axis.legend(loc="best")
    save_figure(fig, output, "figure_c_true_vs_predicted_h56.png")

    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    rescue = rescue_mla_rows[
        rescue_mla_rows.policy_role.eq("Aggressive")
        & rescue_mla_rows.verifier.eq("head_dynamic_abs_w8_b64_r0.1")
    ].output_relative_l2.dropna().sort_values().to_numpy()
    axis.plot([0, 0], [0, 1], label="Full64", color="#111827")
    axis.plot(rescue, np.arange(1, rescue.size + 1) / rescue.size, label="H8 + 10% H56 rescue", color="#2563EB")
    axis.text(
        0.98,
        0.08,
        "No-H56 MLA vectors are not stored.\nNew curves require a GPU follow-up.",
        ha="right",
        va="bottom",
        transform=axis.transAxes,
        bbox={"facecolor": "#FEF3C7", "edgecolor": "#D97706", "alpha": 0.9},
    )
    axis.set(title="Figure D — MLA RelL2 CDF (available measurements only)", xlabel="MLA output relative L2", ylabel="CDF", xlim=(0, min(0.25, max(0.05, float(np.quantile(rescue, 0.995))))))
    axis.legend(loc="upper right")
    save_figure(fig, output, "figure_d_mla_rell2_cdf.png")

    plot_rows = main[main.method.isin([
        "Full64 oracle",
        "Existing H8 + 10% H56 rescue",
        "Raw H8",
        "Per-layer T1 H8 + previous Full64",
        "Hybrid T1 H8 + previous Full64",
    ])]
    fig, axis = plt.subplots(figsize=(7.6, 5.5))
    axis.scatter(100 * plot_rows.qk_reduction, 100 * plot_rows.top2048_recall, s=70)
    for row in plot_rows.itertuples():
        axis.annotate(row.method.replace(" + previous Full64", ""), (100 * row.qk_reduction, 100 * row.top2048_recall), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set(title="Figure E — Selection recall vs indexer QK reduction", xlabel="QK/MAC reduction (%)", ylabel="Top-2048 recall (%)")
    axis.text(0.01, 0.02, "Selection quality only; not MLA/task quality", transform=axis.transAxes, fontsize=9, color="#9A3412")
    axis.grid(alpha=0.25)
    save_figure(fig, output, "figure_e_quality_vs_qk_reduction.png")


def pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the offline H8 reconstruction evidence bundle")
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--offline-root", type=Path, required=True)
    parser.add_argument("--global-rescue-rows", type=Path, required=True)
    parser.add_argument("--global-rescue-mla-summary", type=Path, required=True)
    parser.add_argument("--global-rescue-mla-rows", type=Path, required=True)
    parser.add_argument("--global-rescue-teacher-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    graph_root = args.output / "figures_h8_reconstruction"

    offline_rows = pd.read_csv(args.offline_root / "selection_rows.csv")
    offline_rows = offline_rows[offline_rows.split.eq("test")]
    rescue_rows = pd.read_csv(args.global_rescue_rows)
    mla_summary = pd.read_csv(args.global_rescue_mla_summary)
    mla = mla_summary[
        mla_summary.policy_role.eq("Aggressive")
        & mla_summary.verifier.eq("head_dynamic_abs_w8_b64_r0.1")
    ].iloc[0]
    teacher = pd.read_csv(args.global_rescue_teacher_summary).iloc[0]
    coefficients = json.loads((args.offline_root / "coefficients.json").read_text())
    selected = json.loads((args.offline_root / "selected_methods.json").read_text())
    main = main_results(offline_rows, rescue_rows, mla, teacher)
    main.to_csv(args.output / "main_results.csv", index=False)
    calibration = calibration_table(coefficients)
    calibration.to_csv(args.output / "calibration_parameters.csv", index=False)
    cost = cost_table()
    cost.to_csv(args.output / "cost_metadata_analysis.csv", index=False)

    copies = {
        args.audit_root / "trace_inventory.csv": "trace_inventory.csv",
        args.cache_root / "cpu_full64_numeric_audit.csv": "cpu_full64_numeric_audit.csv",
        args.offline_root / "selection_summary.csv": "selection_summary.csv",
        args.offline_root / "selection_breakdown.csv": "selection_breakdown.csv",
        args.offline_root / "correlation_results.csv": "correlation_results.csv",
        args.offline_root / "residual_statistics.csv": "residual_statistics.csv",
        args.offline_root / "newly_active_summary.csv": "newly_active_summary.csv",
    }
    for source, name in copies.items():
        shutil.copy2(source, args.output / name)
    shutil.copy2(args.offline_root / "coefficients.json", args.output / "coefficients.json")
    shutil.copy2(args.audit_root / "trace_setup.json", args.output / "trace_setup.json")
    shutil.copy2(args.audit_root / "split_manifest.json", args.output / "split_manifest.json")

    figures(
        args.offline_root / "diagnostic_samples.npz",
        main,
        coefficients,
        pd.read_csv(args.global_rescue_mla_rows),
        graph_root,
    )

    pure = main[main.method.eq("Per-layer T1 H8 + previous Full64")].iloc[0]
    hybrid = main[main.method.eq("Hybrid T1 H8 + previous Full64")].iloc[0]
    rescue = main[main.method.eq("Existing H8 + 10% H56 rescue")].iloc[0]
    correlations = pd.read_csv(args.output / "correlation_results.csv")
    newly = pd.read_csv(args.output / "newly_active_summary.csv")
    hybrid_new = newly[newly.method.eq("hybrid_t1_layer")].iloc[0]
    residual = pd.read_csv(args.output / "residual_statistics.csv")
    coefficients_table = "\n".join(
        f"| {int(row.layer)} | {row.t1_a_h8:.6f} | {row.t1_b_previous_full:.6f} | {row.t1_c_intercept:.6f} |"
        for row in calibration.itertuples()
    )
    report = f"""# H8 Full-Score Reconstruction Without H56

## 1. Executive Verdict

**MIXED**

1. **Can H56 be removed entirely?** Not with the tested scalar reconstruction. The validation-selected all-token no-H56 T1 retained only **{pct(pure.top2048_recall)}** of Full64 Top-2048 on the locked test set.
2. **Best no-H56 reconstruction:** per-layer `T1: a_l * H8_t,s + b_l * Full64_(t-1,s) + c_l`. It improved raw H8 ranking but did not recover the broad Full64 ordering.
3. **Quality lost versus H8 + 10% H56 rescue:** Top-2048 recall fell from **{pct(rescue.top2048_recall, 3)}** to **{pct(pure.top2048_recall, 3)}** for complete removal. A deployment-style hybrid that keeps temporal-hot Full64 and removes cold-path H56 reached **{pct(hybrid.top2048_recall, 3)}**, still {100*(rescue.top2048_recall-hybrid.top2048_recall):.3f} percentage points lower. New MLA/LM quality is not present in the offline traces.
4. **QK/MAC reduction:** **87.5%** for complete no-H56 reconstruction. The hybrid achieved **{pct(hybrid.qk_reduction)}** on the same first-64-step test comparison.
5. **Dominant failure mode:** newly-active temporal-cold tokens and the 32K tail. Hybrid Top-2048 recall falls to 97.20% at 32K; newly-active cold-token recall is only **{pct(hybrid_new.newly_active_cold_recall)}**.

Recommendation: **B. Use reconstruction for most blocks, retain a tiny H56 fallback.** Complete H56 removal is not supported; the hybrid result is promising enough for a smaller tail-targeted fallback experiment.

## 2. Dataset / Trace Setup

- Model: `deepseek-ai/DeepSeek-V2-Lite-Chat` research sidecar, revision `85864749cd611b4353ce1decdb286193298f64c7`.
- Full64 score roots: `artifacts/pilot/scores_a/traces`, `artifacts/pilot/scores_b/traces`; calibration: `artifacts/pilot/indexers_1000/traces`.
- Shapes: calibration `(64, 8256)`; heldout `(128, 8320)`, `(128, 16512)`, `(128, 32896)`.
- Layers: 2, 5, 8, 11, 14, 17, 21, 25. Contexts: 8K/16K/32K. B64 and Top-2048 are unchanged.
- Split: 24 locked prior validation traces for calibration; 48 sequence-level validation traces (`code_heldout_4`, `text_heldout_27454`); 96 final test traces from the remaining four heldout sequences.
- Fit observations: 12,434,688 token pairs. Final comparison uses the first 64 transitions because the locked global-budget H8+10% H56 baseline stores 64 transitions.
- H8 was reconstructed from existing hidden captures and frozen sidecar checkpoints on CPU. A 16-row Full64 numerical audit gave Pearson >0.999988 and Top-2048 recall 99.71–99.90% versus the stored GPU trace, quantifying CPU/GPU BF16 variation.
- No model forward, CUDA kernel, or new GPU inference was run.

## 3. Correlation and Residual Structure

Across 233,826,304 valid test observations:

- H8 vs Full64 Pearson/Spearman: **0.8912 / 0.9190**.
- previous Full64 vs current Full64: **0.8520 / 0.8943**.
- H8 vs true H56 residual: **0.4938 / 0.6141**.
- previous Full64 vs current H56 residual: **0.7323 / 0.7918**.
- In temporal-cold regions, H8 vs H56 residual Pearson falls to **0.3634**, while previous Full64 vs residual is **0.6491**.

The omitted residual has global mean/std -2.142/1.305. For newly-active tokens its sampled p95/p99/p99.9 are 0.541/1.478/2.351. The residual is correlated enough to improve ranking but not predictable enough for a tail-safe scalar replacement.

![Figure A](figures_h8_reconstruction/figure_a_h8_vs_full64.png)

![Figure B](figures_h8_reconstruction/figure_b_previous_vs_current_full64.png)

![Figure C](figures_h8_reconstruction/figure_c_true_vs_predicted_h56.png)

## 4. Main Reconstruction Results

| Method | H56 computed? | QK reduction | Top-2048 recall | Top-128 recall | Top-512 recall | RelL2 p95 | RelL2 p99 | KL mean | PPL Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full64 oracle | all 64 heads | 0% | 100% | 100% | 100% | 0% | 0% | 0 | 0% |
| Existing H8 + 10% H56 rescue | cold rescue | {pct(rescue.qk_reduction)} | {pct(rescue.top2048_recall,3)} | {pct(rescue.top128_recall,4)} | {pct(rescue.top512_recall,4)} | {pct(rescue.mla_relative_l2_p95)} | {pct(rescue.mla_relative_l2_p99)} | {rescue.logit_kl_mean:.6f} | {pct(rescue.ppl_delta,3)} |
| Raw H8, all tokens | no | 87.5% | {pct(main[main.method.eq('Raw H8')].iloc[0].top2048_recall)} | {pct(main[main.method.eq('Raw H8')].iloc[0].top128_recall)} | {pct(main[main.method.eq('Raw H8')].iloc[0].top512_recall)} | GPU follow-up | GPU follow-up | GPU follow-up | GPU follow-up |
| Previous Full64 only | no | 100% current QK | {pct(main[main.method.eq('Previous Full64 only')].iloc[0].top2048_recall)} | {pct(main[main.method.eq('Previous Full64 only')].iloc[0].top128_recall)} | {pct(main[main.method.eq('Previous Full64 only')].iloc[0].top512_recall)} | GPU follow-up | GPU follow-up | GPU follow-up | GPU follow-up |
| Per-layer affine H8 | no | 87.5% | {pct(main[main.method.eq('Per-layer affine H8')].iloc[0].top2048_recall)} | {pct(main[main.method.eq('Per-layer affine H8')].iloc[0].top128_recall)} | {pct(main[main.method.eq('Per-layer affine H8')].iloc[0].top512_recall)} | GPU follow-up | GPU follow-up | GPU follow-up | GPU follow-up |
| Per-layer T1 | no | 87.5% | {pct(pure.top2048_recall)} | {pct(pure.top128_recall)} | {pct(pure.top512_recall)} | GPU follow-up | GPU follow-up | GPU follow-up | GPU follow-up |
| Hybrid T1 | cold only | {pct(hybrid.qk_reduction)} | {pct(hybrid.top2048_recall)} | {pct(hybrid.top128_recall)} | {pct(hybrid.top512_recall)} | GPU follow-up | GPU follow-up | GPU follow-up | GPU follow-up |

Positive affine rescaling does not change ranking when the whole score population is H8-derived; raw/global/per-layer/mass-normalized H8 therefore have identical pure Top-K sets. Affine calibration matters only when reconstructed cold scores compete with temporal-hot Full64 scores.

Validation selected these per-layer T1 coefficients, with no test fitting:

| Layer | a (H8) | b (previous Full64) | c |
| ---: | ---: | ---: | ---: |
{coefficients_table}

## 5. Newly-Active Failure Analysis

- Test newly-active observations: **6,691,323 token occurrences** across **1,671,431 B64 block occurrences**.
- Hybrid T1 newly-active recall: **{pct(hybrid_new.newly_active_recall)}** overall, but **{pct(hybrid_new.newly_active_cold_recall)}** for the 248,352 occurrences that remained temporal-cold.
- Hybrid false-negative rate across newly-active tokens: **{pct(hybrid_new.false_negative_rate)}**; worst missed Full64 threshold margin: **{hybrid_new.missed_full64_margin_max:.4f}**.
- Cold-population rank p50/p95 for newly-active cold tokens: **{hybrid_new.cold_rank_p50:.0f} / {hybrid_new.cold_rank_p95:.1f}**.
- The diagnostic `small H8 + top-decile H56 residual` condition occurred 88 times. Most were rescued by temporal-hot treatment, which explains why complete H56 removal degrades more severely than hybrid removal.

## 6. Cost / Metadata Analysis

- Full64 indexer work: `64*128 = 8192` head-dimension MACs per KV token; H8: `8*128 = 1024`, a theoretical **87.5% QK/MAC reduction**.
- T1 reconstruction adds roughly five scalar FLOPs per token (`a*h8 + b*prev + c`).
- H8 still reads the shared 128-D K vector. At BF16 this is **256 bytes/token**, so no physical K-byte reduction is claimed.
- Previous-score metadata is 4 bytes/token in FP32 or 2 bytes/token in FP16/BF16/INT16. Relative to a 256-byte BF16 indexer K it is 1.5625% or 0.78125% per layer.
- At 128K, previous-score state is 512 KiB/layer in FP32 and 256 KiB/layer in FP16; across eight evaluated layers this is 4 MiB or 2 MiB.

## 7. Comparison to Existing H8 + 10% Rescue

The locked global-budget rescue baseline keeps Top-2048 recall at {pct(rescue.top2048_recall,3)} with median QK reduction {pct(rescue.qk_reduction)} and measured MLA RelL2 p95/p99 {pct(rescue.mla_relative_l2_p95)}/{pct(rescue.mla_relative_l2_p99)}. Complete no-H56 T1 gains another {100*(pure.qk_reduction-rescue.qk_reduction):.2f} percentage points of theoretical QK reduction but loses {100*(rescue.top2048_recall-pure.top2048_recall):.2f} recall points. Hybrid T1 gains {100*(hybrid.qk_reduction-rescue.qk_reduction):.2f} QK-reduction points and loses {100*(rescue.top2048_recall-hybrid.top2048_recall):.3f} recall points.

The new sparse selections require the main-attention V tensors/projection and model logits, which are not stored in the offline score trace. Therefore:

- **OFFLINE TRACE RESULT:** all correlation, Top-K, newly-active, attention-mass proxy, cost, and split-controlled regression results in this report.
- **GPU FOLLOW-UP REQUIRED:** new-policy MLA RelL2/cosine, logit KL, and PPL. No GPU work was launched automatically.

![Figure D](figures_h8_reconstruction/figure_d_mla_rell2_cdf.png)

![Figure E](figures_h8_reconstruction/figure_e_quality_vs_qk_reduction.png)

## 8. Recommendation

### B. Use reconstruction for most blocks, retain tiny H56 fallback

Do not replace all current-step H56 with scalar reconstruction. The next experiment should keep T1 for ordinary cold blocks and invoke H56 only for a validation-calibrated tail trigger targeting newly-active cold tokens near the Top-2048 boundary. A useful target is below 1–2% of cold blocks, followed by the deferred MLA and LM-quality replay.

```text
Best no-H56 policy: Per-layer T1 reconstruction
Formula: I_hat[t,s] = a_l * H8[t,s] + b_l * Full64[t-1,s] + c_l
Calibration parameters: 8 per-layer (a,b,c) tuples in calibration_parameters.csv
QK reduction: 87.5%
Top-2048 recall: {pct(pure.top2048_recall,3)}
Top-128 recall: {pct(pure.top128_recall,3)}
Top-512 recall: {pct(pure.top512_recall,3)}
MLA RelL2 p95 / p99: GPU FOLLOW-UP REQUIRED
Logit KL: GPU FOLLOW-UP REQUIRED
PPL delta: GPU FOLLOW-UP REQUIRED

vs H8 + 10% H56 rescue:
Quality change: Top-2048 recall {100*(pure.top2048_recall-rescue.top2048_recall):+.3f} percentage points; MLA/LM change pending
Compute change: QK reduction {pct(rescue.qk_reduction)} -> 87.5%
Metadata change: add one previous-score scalar per KV token/layer (2 or 4 bytes)

Main remaining failure: newly-active temporal-cold tokens, especially at 32K
Recommended next experiment: per-layer T1 plus <1–2% validation-fixed tail H56 fallback, then MLA/KL/PPL replay
```
"""
    (args.output / "h8_full_score_reconstruction_report.md").write_text(report, encoding="utf-8")

    verdict = {
        "verdict": "MIXED",
        "can_remove_h56_entirely": False,
        "recommendation": "B_use_reconstruction_for_most_blocks_retain_tiny_h56_fallback",
        "best_no_h56": {
            "method": selected["best_pure_temporal"],
            "formula": "a_l*h8 + b_l*previous_full + c_l",
            "qk_reduction": float(pure.qk_reduction),
            "top2048_recall": float(pure.top2048_recall),
            "top128_recall": float(pure.top128_recall),
            "top512_recall": float(pure.top512_recall),
        },
        "best_hybrid": {
            "method": selected["best_hybrid_temporal"],
            "qk_reduction": float(hybrid.qk_reduction),
            "top2048_recall": float(hybrid.top2048_recall),
            "top128_recall": float(hybrid.top128_recall),
            "top512_recall": float(hybrid.top512_recall),
        },
        "existing_global_budget_rescue": {
            "qk_reduction": float(rescue.qk_reduction),
            "top2048_recall": float(rescue.top2048_recall),
            "mla_relative_l2_p95": float(rescue.mla_relative_l2_p95),
            "mla_relative_l2_p99": float(rescue.mla_relative_l2_p99),
            "logit_kl_mean": float(rescue.logit_kl_mean),
            "ppl_delta": float(rescue.ppl_delta),
        },
        "offline_only": True,
        "gpu_used": False,
        "gpu_followup_required": ["MLA output RelL2/cosine", "logit KL", "PPL"],
    }
    (args.output / "h8_reconstruction_verdict.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reproducibility = {
        "trace_audit": "scripts/audit_h8_reconstruction_traces.py",
        "h8_cache": "scripts/cache_h8_reconstruction_scores.py with CUDA_VISIBLE_DEVICES empty",
        "offline_replay": "scripts/evaluate_h8_reconstruction_offline.py with CUDA_VISIBLE_DEVICES empty",
        "report": "scripts/build_h8_reconstruction_report.py",
        "calibration_fit_uses_test": False,
        "validation_selects_method": True,
        "test_fit_or_selection": False,
        "global_baseline_transition_count": 64,
        "gpu_used": False,
    }
    (args.output / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
