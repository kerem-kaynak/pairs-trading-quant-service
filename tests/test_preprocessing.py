import pytest
import pandas as pd
import pendulum
from utils.preprocessing import compute_returns, construct_df_from_ohlc

def test_compute_returns():
    df = pd.DataFrame({
        'A': [100, 110, 105, 115],
        'B': [50, 55, 52, 58]
    }, index=pd.date_range('2023-01-01', periods=4))
    result = compute_returns(df)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert result.iloc[0]['A'] == pytest.approx(0.1)
    assert result.iloc[1]['B'] == pytest.approx(-0.05454545, rel=1e-5)

def test_construct_df_from_ohlc():
    data = [
        {"ticker": "AAPL", "date": "2023-01-01", "price": 100},
        {"ticker": "AAPL", "date": "2023-01-02", "price": 105},
        {"ticker": "GOOGL", "date": "2023-01-01", "price": 200},
        {"ticker": "GOOGL", "date": "2023-01-03", "price": 210}
    ]
    result = construct_df_from_ohlc(data)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 2)
    assert result.index.name == 'date'
    assert result.loc['2023-01-02', 'GOOGL'] == pytest.approx(205.0)