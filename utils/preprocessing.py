from typing import Any, Dict, List
import numpy as np
import pandas as pd

def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a pandas DataFrame of daily returns.

    :param df: A pandas DataFrame of closing prices of tickers
    :returns: A pandas DataFrame where the dates are the index, column names are the tickers and the values are the daily returns
    """

    df_returns = df.pct_change().iloc[1:]
    df_returns = df_returns.replace([np.inf, -np.inf], 0)
    df_returns = df_returns.loc[:, (df_returns != 0).any(axis=0)]
    return df_returns

def construct_df_from_ohlc(
    data: List[Dict[str, Any]],
    date_column_key: str = "date",
    ticker_column_key: str = "ticker",
    price_column_key: str = "price"
) -> pd.DataFrame:
    """
    Constructs a pandas DataFrame from a list of OHLC (Open, High, Low, Close) dictionaries.

    :param data: Input data for the DataFrame
    :param date_column_key: Key for the value corresponding to the date in each dictionary in the input
    :param ticker_column_key: Key for the value corresponding to the ticker in each dictionary in the input
    :param price_column_key: Key for the value corresponding to the price in each dictionary in the input
    :returns: A pandas DataFrame where the dates are the index, column names are the tickers and the values are the prices
    """
    df = pd.DataFrame(data)
    df[date_column_key] = pd.to_datetime(df[date_column_key])

    pivot_df = df.pivot_table(index=date_column_key, columns=ticker_column_key, values=price_column_key, aggfunc="first")
    pivot_df.sort_index(inplace=True)
    pivot_df = pivot_df.astype(float)
    pivot_df = pivot_df.interpolate(method='linear')
    valid_columns = pivot_df.columns[pivot_df.iloc[0].notna() & pivot_df.iloc[-1].notna()]
    pivot_df = pivot_df[valid_columns]

    return pivot_df