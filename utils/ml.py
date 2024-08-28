from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Union
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS
import pendulum


def apply_pca_and_scaling(df_returns: pd.DataFrame) -> np.ndarray:
    """
    Apply PCA and scaling to the input DataFrame of returns.

    :param df_returns: DataFrame of returns
    :return: Scaled principal components
    """
    if df_returns.shape[1] < 2:
        raise ValueError("Not enough features for PCA. Need at least 2 columns with non-constant values.")

    n_components = min(5, df_returns.shape[1] - 1)
    pca = PCA(n_components=n_components, svd_solver='auto', random_state=42)
    pca.fit(df_returns)

    scaler = StandardScaler()
    scaled_principal_components = scaler.fit_transform(pca.components_.T)
    
    return scaled_principal_components

def apply_optics(scaled_principal_components: np.ndarray, df_returns: pd.DataFrame) -> List[List[str]]:
    """
    Apply OPTICS clustering to the scaled principal components and generate pairs to evaluate.

    :param scaled_principal_components: Scaled principal components from PCA
    :param df_returns: DataFrame of returns
    :return: List of pairs to evaluate
    """
    optics = OPTICS(min_samples=5, max_eps=10, xi=0.1, metric='euclidean', cluster_method='xi')
    optics.fit(scaled_principal_components)
    labels = optics.labels_

    clustered_series_all = pd.Series(index=df_returns.columns, data=labels.flatten())
    clustered_series = clustered_series_all[clustered_series_all != -1]
    cluster_dict = defaultdict(list)
    
    pairs_dict = defaultdict(list)
    pairs_to_eval = []
    
    for i in range(len(clustered_series)):
        cluster_dict[int(clustered_series.iloc[i])].append(clustered_series.index[i])
    
    for k, v in cluster_dict.items():
        pair_combinations = list(combinations(v, 2))
        pairs_dict[k] = pair_combinations
    
    for k, v in pairs_dict.items():
        pairs_to_eval += pairs_dict[k]

    return pairs_to_eval

def calculate_rlrt_trend_and_confidence(dates: List[pendulum.DateTime], spreads: List[float]) -> Dict[str, Union[str, float]]:
    """
    Calculate the trend and confidence for a given set of dates and spreads.

    :param dates: List of dates for the calculation
    :param spreads: List of spread values corresponding to the dates
    :returns: A dictionary containing the date, trend, and confidence
    """
    x = np.array([(date - dates[0]).total_seconds() / 86400 for date in dates])
    y = np.array(spreads)
    
    slope, intercept, r_value, _, _ = linregress(x, y)
    r2 = r_value ** 2    
    
    next_day = len(x)
    prediction = slope * next_day + intercept
    trend = "positive" if prediction > spreads[-1] else "negative"
    
    formatted_date = dates[-1].format('YYYY-MM-DD')
    
    return {
        "date": formatted_date,
        "trend": trend,
        "confidence": r2
    }