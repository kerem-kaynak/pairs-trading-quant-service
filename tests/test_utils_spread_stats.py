import pytest
import pandas as pd
import numpy as np
from utils.spread_stats import (
    compute_spread_statistics,
    compute_cointegration_critical_value,
    compute_hurst_exponent,
    compute_half_life,
    calculate_mean_crossing_frequency,
    run_statistical_criteria_tests_for_pairs
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': np.random.rand(100) * 100,
        'B': np.random.rand(100) * 50
    }, index=pd.date_range('2023-01-01', periods=100))

def test_compute_spread_statistics(sample_df):
    slope, intercept, residuals = compute_spread_statistics(sample_df, 'A', 'B')
    assert isinstance(slope, float)
    assert isinstance(intercept, float)
    assert isinstance(residuals, pd.Series)

def test_compute_cointegration_critical_value(sample_df):
    _, _, residuals = compute_spread_statistics(sample_df, 'A', 'B')
    critical_value = compute_cointegration_critical_value(residuals)
    assert isinstance(critical_value, float)

def test_compute_hurst_exponent(sample_df):
    _, _, residuals = compute_spread_statistics(sample_df, 'A', 'B')
    hurst = compute_hurst_exponent(residuals)
    assert isinstance(hurst, float)
    assert 0 <= hurst <= 1

def test_compute_half_life(sample_df):
    _, _, residuals = compute_spread_statistics(sample_df, 'A', 'B')
    half_life = compute_half_life(residuals)
    assert isinstance(half_life, float)

def test_calculate_mean_crossing_frequency(sample_df):
    _, _, residuals = compute_spread_statistics(sample_df, 'A', 'B')
    mcf = calculate_mean_crossing_frequency(residuals)
    assert isinstance(mcf, int)

def test_run_statistical_criteria_tests_for_pairs(sample_df):
    pairs_to_eval = [('A', 'B')]
    result = run_statistical_criteria_tests_for_pairs(pairs_to_eval, sample_df)
    assert isinstance(result, list)
    if result:
        assert isinstance(result[0], dict)
        assert 'ticker_1' in result[0]
        assert 'ticker_2' in result[0]