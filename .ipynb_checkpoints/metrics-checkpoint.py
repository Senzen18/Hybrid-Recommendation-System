import numpy as np
from collections.abc import Callable
import pandas as pd
def dcg_score (y_true, y_score, k = 20, gains = "exponential"):
  
    order = np.argsort (y_score) [::-1]
    y_true = np.take (y_true, order [: k])

    if gains == "exponential":
        gains = 2 ** y_true-1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError ("Invalid gains option.")

    discounts = np.log2 (np.arange (len (y_true)) + 2)
    return np.sum (gains / discounts)

def ndcg_score (y_true, y_score, k = 20, gains = "exponential"):
    best = dcg_score (y_true, y_true, k, gains)
    actual = dcg_score (y_true, y_score, k, gains)
    return actual / best


def check_inputs(func) -> Callable:
    """
    Decorator function to validate inputs to precision_at_k & recall_at_k
    """
    def checker(df: pd.DataFrame, k: int=3, y_test: str='y_actual', y_pred: str='y_recommended') -> float:
       
        if k <= 0:
            raise ValueError(f'Value of k should be greater than 1, read in as: {k}')
       
        if y_test not in df.columns:
            raise ValueError(f'Input dataframe does not have a column named: {y_test}')
        if y_pred not in df.columns:
            raise ValueError(f'Input dataframe does not have a column named: {y_pred}')
        return func(df, k, y_test, y_pred)
    return checker

@check_inputs
def precision_at_k(df: pd.DataFrame, k: int, y_test: str, y_pred: str) -> float:
    
    dfK = df.head(k)
    denominator = dfK[y_pred].sum()
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    if denominator > 0:
        return numerator/denominator
    else:
        return None

@check_inputs
def recall_at_k(df: pd.DataFrame, k: int, y_test: str, y_pred: str) -> float:
  
    dfK = df.head(k)
    denominator = df[y_test].sum()
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    if denominator > 0:
        return numerator/denominator
    else:
        return None