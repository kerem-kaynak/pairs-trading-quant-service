import pytest
import pandas as pd
import numpy as np
from utils.trading import rolling_regression_trend_with_confidence, trade_pair_using_model

def test_rolling_regression_trend_with_confidence():
    data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4, 5])
    trends, confidences = rolling_regression_trend_with_confidence(data, window_size=5, forecast_days=1)
    
    assert len(trends) == 8
    assert len(confidences) == 8
    assert all(trend in [-1, 0, 1] for trend in trends)
    assert all(0 <= conf <= 1 for conf in confidences)

def test_trade_pair_using_model():
    dates = pd.date_range(start='2020-01-01', periods=100)
    df = pd.DataFrame({
        'ticker1': np.random.randn(100).cumsum(),
        'ticker2': np.random.randn(100).cumsum()
    }, index=dates)

    result = trade_pair_using_model(df, 'ticker1', 'ticker2')

    assert 'results' in result
    assert 'max_drawdown' in result
    assert 'total_return' in result
    assert 'annualized_return' in result

    assert len(result['results']) == 100
    assert isinstance(result['max_drawdown'], float)
    assert isinstance(result['total_return'], float)
    assert isinstance(result['annualized_return'], float)

def test_rolling_regression_trend_with_confidence_edge_cases():
    data = np.array([1, 2, 3])
    trends, confidences = rolling_regression_trend_with_confidence(data, window_size=2, forecast_days=1)
    assert len(trends) == 1
    assert len(confidences) == 1

    data = np.array([1, 1, 1, 1, 1])
    trends, confidences = rolling_regression_trend_with_confidence(data, window_size=2, forecast_days=1)
    assert all(trend == 0 for trend in trends)

def test_trade_pair_using_model_edge_cases():
    df = pd.DataFrame()
    with pytest.raises(KeyError):
        trade_pair_using_model(df, 'ticker1', 'ticker2')

    df = pd.DataFrame({'ticker1': [1], 'ticker2': [2]}, index=[pd.Timestamp('2020-01-01')])
    result = trade_pair_using_model(df, 'ticker1', 'ticker2')
    assert len(result['results']) == 1
    assert result['max_drawdown'] == 0
    assert result['total_return'] == 0
    assert result['annualized_return'] == 0

    dates = pd.date_range(start='2020-01-01', periods=100)
    df = pd.DataFrame({
        'ticker1': np.arange(100),
        'ticker2': np.arange(100)
    }, index=dates)
    result = trade_pair_using_model(df, 'ticker1', 'ticker2')
    assert len(result['results']) == 100
    assert result['max_drawdown'] == 0
    assert result['total_return'] == 0
    assert result['annualized_return'] == 0