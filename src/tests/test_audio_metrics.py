"""Unit tests for the set-level audio statistics (no model download needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "evaluation"))
from audio_metrics import fad_bootstrap, frechet_distance, kl_divergence  # noqa: E402


def test_frechet_distance_is_zero_for_the_same_set_and_grows_with_a_mean_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2000, 8))
    assert frechet_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    shift = np.zeros(8)
    shift[0] = 3.0
    assert frechet_distance(a, a + shift) == pytest.approx(9.0, rel=1e-3)  # |mu_a - mu_b|^2, same covariance


def test_fad_bootstrap_is_reproducible_and_smaller_for_the_closer_set():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(500, 6))
    near, far = rng.normal(size=(300, 6)), rng.normal(loc=1.0, size=(300, 6))
    r1 = fad_bootstrap(ref, near, 100, 5, np.random.default_rng(7))
    r2 = fad_bootstrap(ref, near, 100, 5, np.random.default_rng(7))
    assert r1 == r2
    assert r1["fad"] < fad_bootstrap(ref, far, 100, 5, np.random.default_rng(7))["fad"]


def test_kl_divergence_of_identical_distributions_is_zero():
    p = np.array([0.2, 0.5, 0.3])
    assert kl_divergence(p, p * 4) == pytest.approx(0.0, abs=1e-9)  # inputs are normalized first
    assert kl_divergence(p, np.array([0.6, 0.2, 0.2])) > 0
