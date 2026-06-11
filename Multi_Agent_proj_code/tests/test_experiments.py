import pytest

from experiments.adaptive_vs_static import two_proportion_ztest
from experiments.transferability import TransferabilityExperiment


def test_ztest_equal_proportions_not_significant():
    z, p = two_proportion_ztest(10, 50, 10, 50)
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_ztest_large_difference_significant():
    z, p = two_proportion_ztest(5, 100, 60, 100)
    assert p < 0.001
    assert z > 0  # second arm (adaptive) higher


def test_ztest_empty_samples_safe():
    assert two_proportion_ztest(0, 0, 0, 0) == (0.0, 1.0)


def test_wilson_interval_bounds():
    lo, hi = TransferabilityExperiment._wilson_interval(0.5, 20)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    # More data → tighter interval
    lo2, hi2 = TransferabilityExperiment._wilson_interval(0.5, 200)
    assert (hi2 - lo2) < (hi - lo)


def test_wilson_interval_degenerate():
    assert TransferabilityExperiment._wilson_interval(0.0, 0) == (0.0, 1.0)
    lo, hi = TransferabilityExperiment._wilson_interval(1.0, 10)
    assert hi == 1.0 and lo < 1.0
