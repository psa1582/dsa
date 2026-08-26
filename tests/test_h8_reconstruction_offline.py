from __future__ import annotations

import numpy as np

from scripts.evaluate_h8_reconstruction_offline import (
    LeastSquares,
    margin_bucket,
    selection_metrics,
)


def test_least_squares_recovers_affine_coefficients() -> None:
    x = np.linspace(-3.0, 3.0, 101)
    y = 1.75 * x - 0.25
    fit = LeastSquares(2)
    fit.update([x, np.ones_like(x)], y)

    assert fit.count == x.size
    assert np.allclose(fit.solve(), [1.75, -0.25], atol=1e-10)


def test_margin_bucket_has_stable_boundary_assignment() -> None:
    previous = np.array([7.9, 8.0, 9.9, 10.0, 10.1, 12.0, 12.1])

    buckets = margin_bucket(previous, tau=10.0, band=2.0)

    assert buckets.tolist() == [3, 2, 2, 1, 1, 1, 0]


def test_selection_metrics_reports_exact_and_single_swap() -> None:
    full = np.linspace(10.0, -10.0, 4096, dtype=np.float32)
    baseline = np.arange(2048, dtype=np.int64)
    perfect = selection_metrics(baseline.copy(), baseline, full, teacher=None)

    assert perfect["exact_top2048_match"]
    assert perfect["top2048_recall"] == 1.0
    assert perfect["false_negative_count"] == 0

    swapped = baseline.copy()
    swapped[-1] = 2048
    imperfect = selection_metrics(swapped, baseline, full, teacher=None)

    assert not imperfect["exact_top2048_match"]
    assert imperfect["top2048_recall"] == 2047 / 2048
    assert imperfect["top2048_precision"] == 2047 / 2048
    assert imperfect["false_negative_count"] == 1
    assert imperfect["false_positive_count"] == 1
