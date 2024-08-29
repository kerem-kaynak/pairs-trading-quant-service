from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


def rolling_regression_trend_with_confidence(
    data: np.ndarray, 
    window_size: int = 10, 
    forecast_days: int = 3, 
    r2_threshold: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute rolling regression trend and confidence for given data.

    :param data: Input data array
    :param window_size: Size of the rolling window
    :param forecast_days: Number of days to forecast
    :param r2_threshold: R-squared threshold for trend determination
    :return: Tuple of trend array and confidence array
    """
    
    trends = []
    confidences = []
    for i in range(window_size, len(data) - forecast_days + 1):
        X = np.arange(window_size).reshape(-1, 1)
        y = data[i-window_size:i]
        model = LinearRegression()
        model.fit(X, y)
        
        r2 = r2_score(y, model.predict(X))
        
        if r2 > r2_threshold:
            trend = 1 if model.coef_[0] > 0 else -1 if model.coef_[0] < 0 else 0
        else:
            trend = 0
        
        trends.append(trend)
        confidences.append(r2)
    return np.array(trends), np.array(confidences)

def trade_pair_using_model(df: pd.DataFrame, ticker_1: str, ticker_2: str) -> Dict[str, Any]:
    """
    Perform pairs trading using RLRT and compute trade statistics.

    :param df: DataFrame containing price data for both tickers
    :param ticker_1: First ticker symbol
    :param ticker_2: Second ticker symbol
    :return: Dictionary containing trading results and statistics
    """

    ticker_series_1 = df[ticker_1]
    ticker_series_2 = df[ticker_2]

    slope, intercept, _, _, _ = linregress(ticker_series_1, ticker_series_2)
    spread = ticker_series_2 - (slope * ticker_series_1 + intercept)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(spread.values.reshape(-1, 1)).reshape(-1)

    predicted_trends, _ = rolling_regression_trend_with_confidence(data, r2_threshold=0.6)

    initial_budget = 100000
    dates = df.index

    window_size = 10

    signals = ['None'] * len(data)
    positions = [0] * len(data)
    budgets = [initial_budget] * len(data)

    cumulative_mean = np.zeros(len(data))
    cumulative_std = np.zeros(len(data))

    padded_predictions = [None] * window_size + list(predicted_trends)
    if len(padded_predictions) < len(data):
        padded_predictions.extend([None] * (len(data) - len(padded_predictions)))
    else:
        padded_predictions = padded_predictions[:len(data)]

    for i in range(len(data)):
        if i < window_size:
            cumulative_mean[i] = np.mean(data[:i+1])
            cumulative_std[i] = np.std(data[:i+1]) if i > 0 else 0
            continue
        
        cumulative_mean[i] = np.mean(data[:i+1])
        cumulative_std[i] = np.std(data[:i+1])
        
        spread_value = data[i]
        trend_prediction = padded_predictions[i] if padded_predictions[i] is not None else 0
        
        if spread_value > cumulative_mean[i] + cumulative_std[i]:
            if trend_prediction <= 0:
                signals[i] = "Short"
            else:
                signals[i] = "None"
        elif spread_value < cumulative_mean[i] - cumulative_std[i]:
            if trend_prediction >= 0:
                signals[i] = "Long"
            else:
                signals[i] = "None"
        else:
            if trend_prediction == 1:
                signals[i] = "Exit Short"
            elif trend_prediction == -1:
                signals[i] = "Exit Long"
            else:
                signals[i] = "None"
        
        current_position = positions[i-1]
        if signals[i] == "Short" and current_position != -1:
            if current_position == 1:
                returns = (ticker_series_2.iloc[i] / ticker_series_2.iloc[i-1] - 1) - \
                        (ticker_series_1.iloc[i] / ticker_series_1.iloc[i-1] - 1)
                budgets[i] = budgets[i-1] * (1 + returns)
            positions[i] = -1
        elif signals[i] == "Long" and current_position != 1:
            if current_position == -1:
                returns = -((ticker_series_2.iloc[i] / ticker_series_2.iloc[i-1] - 1) - \
                            (ticker_series_1.iloc[i] / ticker_series_1.iloc[i-1] - 1))
                budgets[i] = budgets[i-1] * (1 + returns)
            positions[i] = 1
        elif (signals[i] == "Exit Short" and current_position == -1) or (signals[i] == "Exit Long" and current_position == 1):
            returns = current_position * ((ticker_series_2.iloc[i] / ticker_series_2.iloc[i-1] - 1) - \
                                        (ticker_series_1.iloc[i] / ticker_series_1.iloc[i-1] - 1))
            budgets[i] = budgets[i-1] * (1 + returns)
            positions[i] = 0
        else:
            positions[i] = current_position
            
        if current_position != 0:
            returns = current_position * ((ticker_series_2.iloc[i] / ticker_series_2.iloc[i-1] - 1) - \
                                        (ticker_series_1.iloc[i] / ticker_series_1.iloc[i-1] - 1))
            budgets[i] = budgets[i-1] * (1 + returns)
        else:
            budgets[i] = budgets[i-1]
    
    date_strings = [date.strftime('%Y-%m-%d') for date in dates]

    results_df = pd.DataFrame({
        'date': date_strings,
        'spread': data,
        'predicted_trend': padded_predictions,
        'signal': signals,
        'position': positions,
        'budget': budgets,
        'ticker_1': ticker_series_1,
        'ticker_2': ticker_series_2
    })

    results_df['daily_return'] = results_df['budget'].pct_change()
    results_df['cumulative_return'] = results_df['budget'] / initial_budget - 1

    roll_max = results_df['budget'].cummax()
    daily_drawdown = results_df['budget'] / roll_max - 1.0
    max_drawdown = daily_drawdown.min()

    start_date = pd.to_datetime(results_df['date'].iloc[0])
    end_date = pd.to_datetime(results_df['date'].iloc[-1])

    years = (end_date - start_date).days / 365.25

    total_return = (budgets[-1] / initial_budget) - 1

    annualized_return = 0 if (years == 0) else ((1 + total_return) ** (1 / years) - 1)

    results_list = results_df.to_dict(orient='records')

    return {
        "results": results_list,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
        "annualized_return": annualized_return
    }