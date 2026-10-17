# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for PerfStats percentile reporting."""

import numpy as np
import pytest

from winml.modelkit.session.stats import PerfStats


def test_p50_is_true_median_for_even_sample_count() -> None:
    stats = PerfStats(_samples=[4.0, 1.0, 3.0, 2.0])

    assert stats.p50_ms == pytest.approx(2.5)


@pytest.mark.parametrize("p", [0, 10, 50, 90, 95, 99, 100])
def test_percentile_matches_numpy_linear_interpolation(p: float) -> None:
    samples = list(np.random.default_rng(7).uniform(1.0, 9.0, 37))
    stats = PerfStats(_samples=samples)

    assert stats.percentile(p) == pytest.approx(float(np.percentile(samples, p)))


def test_percentile_excludes_warmup_samples() -> None:
    stats = PerfStats(_samples=[100.0, 1.0, 2.0, 3.0], warmup=1)

    assert stats.p50_ms == pytest.approx(2.0)
    assert stats.percentile(100) == pytest.approx(3.0)
