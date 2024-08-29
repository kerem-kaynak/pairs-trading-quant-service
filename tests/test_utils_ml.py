import pytest
import numpy as np
import pandas as pd
import pendulum
from utils.ml import apply_pca_and_scaling, apply_optics, calculate_rlrt_trend_and_confidence

def test_apply_pca_and_scaling():
    df_returns = pd.DataFrame(np.random.rand(100, 10))
    result = apply_pca_and_scaling(df_returns)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 5 

def test_apply_optics():
    scaled_principal_components = np.random.rand(10, 5)
    df_returns = pd.DataFrame(np.random.rand(100, 10), columns=[f'ticker_{i}' for i in range(10)])
    result = apply_optics(scaled_principal_components, df_returns)
    assert isinstance(result, list)
    for pair in result:
        assert len(pair) == 2
        assert isinstance(pair[0], str)
        assert isinstance(pair[1], str)

def test_calculate_rlrt_trend_and_confidence():
    dates = [pendulum.now().add(days=i) for i in range(10)]
    spreads = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    result = calculate_rlrt_trend_and_confidence(dates, spreads)
    assert isinstance(result, dict)
    assert 'date' in result
    assert 'trend' in result
    assert 'confidence' in result
    assert result['trend'] in ['positive', 'negative']
    assert 0 <= result['confidence'] <= 1