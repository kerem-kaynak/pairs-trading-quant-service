import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
from hurst import compute_Hc
from typing import Tuple, List, Dict, Any


def compute_spread_statistics(df: pd.DataFrame, ticker_1: str, ticker_2: str) -> Tuple[float, float, pd.Series]:
    """
    Compute spread statistics for a pair of tickers.

    :param df: DataFrame containing price data
    :param ticker_1: First ticker symbol
    :param ticker_2: Second ticker symbol
    :return: Tuple of slope, intercept, and residuals
    """
    price_series_1 = df[ticker_1]
    price_series_2 = df[ticker_2]

    slope, intercept, _, _, _ = linregress(price_series_1, price_series_2)
    residuals = price_series_2 - (slope * price_series_1 + intercept)

    return slope, intercept, residuals

def compute_cointegration_critical_value(residuals: pd.Series) -> float:
    """
    Compute the cointegration critical value for the given residuals.

    :param residuals: Series of residuals
    :return: Cointegration critical value
    """
    cointegration_result = adfuller(residuals, autolag='AIC')

    return cointegration_result[1]

def compute_hurst_exponent(residuals: pd.Series) -> float:
    """
    Compute the Hurst exponent for the given residuals.

    :param residuals: Series of residuals
    :return: Hurst exponent
    """
    H_val, _, _ = compute_Hc(residuals)

    return H_val

def compute_half_life(residuals: pd.Series) -> float:
    """
    Compute the half-life of mean reversion for the given residuals.

    :param residuals: Series of residuals
    :return: Half-life of mean reversion
    """
    lagged_residuals = np.roll(residuals, 1)
    lagged_residuals[0] = 0
    
    delta_residuals = residuals - lagged_residuals
    lagged_residuals_with_intercept = np.vstack([lagged_residuals, np.ones(len(lagged_residuals))]).T
    
    model = OLS(delta_residuals, lagged_residuals_with_intercept).fit()
    
    half_life = -np.log(2) / model.params.iloc[0]

    return half_life

def calculate_mean_crossing_frequency(residuals: pd.Series) -> int:
    """
    Calculate the frequency of mean crossings for the given residuals.

    :param residuals: Series of residuals
    :return: Number of mean crossings
    """
    delta_residuals_mean = residuals - np.mean(residuals)
    mean_crossings = sum(1 for i, _ in enumerate(delta_residuals_mean) if (i + 1 < len(delta_residuals_mean)) if ((delta_residuals_mean.iloc[i] * delta_residuals_mean.iloc[i + 1] < 0) or (delta_residuals_mean.iloc[i] == 0)))

    return mean_crossings

def run_statistical_criteria_tests_for_pairs(
        pairs_to_eval: List[Tuple[str, str]],
        df: pd.DataFrame,
        cointegration_threshold: float = 0.1,
        hurst_exponent_threshold: float = 0.5,
        half_life_threshold: float = 260,
        mean_crossings_threshold: int = 8
    ) -> List[Dict[str, Any]]:
    """
    Run statistical criteria tests for the given pairs and return valid pairs.

    :param pairs_to_eval: List of ticker pairs to evaluate
    :param df: DataFrame containing price data
    :param cointegration_threshold: Threshold for cointegration test
    :param hurst_exponent_threshold: Threshold for Hurst exponent
    :param half_life_threshold: Threshold for half-life of mean reversion
    :param mean_crossings_threshold: Threshold for mean crossing frequency
    :return: List of dictionaries containing valid pairs and their statistics
    """
    criteria_valid_pairs = []
    for pair in pairs_to_eval:
        slope, intercept, residuals = compute_spread_statistics(df, pair[0], pair[1])
        cointegration_critical_value = compute_cointegration_critical_value(residuals)
        hurst_exponent = compute_hurst_exponent(residuals)
        half_life = compute_half_life(residuals)
        mean_crossings = calculate_mean_crossing_frequency(residuals)

        if cointegration_critical_value < cointegration_threshold and hurst_exponent < hurst_exponent_threshold and half_life < half_life_threshold and mean_crossings > mean_crossings_threshold:
            criteria_valid_pairs.append({
                "ticker_1": pair[0],
                "ticker_2": pair[1],
                "spread_statistics": {
                    "slope": slope,
                    "intercept": intercept,
                },
                "cointegration_critical_value": cointegration_critical_value,
                "hurst_exponent": hurst_exponent,
                "half_life": half_life,
                "mean_crossings": mean_crossings
            })
    
    return criteria_valid_pairs

def compute_statistical_criteria_tests_for_pair(
        pair: List[str],
        df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
    """
    Compute statistical criteria for the given pair.

    :param pair: List of tickers that make up the pair to test
    :param df: DataFrame containing price data
    :return: Dictionary containing test results
    """
    slope, intercept, residuals = compute_spread_statistics(df, pair[0], pair[1])
    cointegration_critical_value = compute_cointegration_critical_value(residuals)
    hurst_exponent = compute_hurst_exponent(residuals)
    half_life = compute_half_life(residuals)
    mean_crossings = calculate_mean_crossing_frequency(residuals)
    result = {
        "ticker_1": pair[0],
        "ticker_2": pair[1],
        "spread_statistics": {
            "slope": slope,
            "intercept": intercept,
            "residuals": residuals
        },
        "cointegration_critical_value": cointegration_critical_value,
        "hurst_exponent": hurst_exponent,
        "half_life": half_life,
        "mean_crossings": mean_crossings
    }
    
    return result