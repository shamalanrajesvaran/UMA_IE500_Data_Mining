import numpy as np
import pandas as pd

def encode(df: pd.DataFrame, method: str, columns: list[str]) -> pd.DataFrame:
    """
    Encode *columns* in *df* using one-hot, frequency, or circular encoding.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data (left unchanged; work is done on a copy).
    method : str
        'onehot' | 'frequency' | 'circular'
        (aliases: 'one-hot', 'ohe', 'freq', 'cyclic', 'sincos').
    columns : list[str]
        List of column names to encode.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing the encoded features.
    """
    m = method.lower()

    if m in {'onehot', 'one-hot', 'one_hot', 'ohe'}:
        df = pd.get_dummies(df, columns=columns, drop_first=True, dtype=int)

    elif m in {'frequency', 'freq'}:
        for col in columns:
            freq_map = df[col].value_counts().to_dict()
            df[f"{col}_frequency_encoded"] = df[col].map(freq_map)
        df.drop(columns=columns, inplace=True)

    elif m in {'circular', 'cyclic', 'sincos'}:
        for col in columns:
            max_val = df[col].max()
            df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
            df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
        df.drop(columns=columns, inplace=True)

    else:
        raise ValueError("method must be 'onehot', 'frequency', or 'circular'")

    return df






ONEHOT_COLS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "assigned_room_type",
    "deposit_type",
    "customer_type",
    "arrival_weekday",
    "booking_date_month",
    "booking_weekday",
    "booking_season",
    "arrival_season",
]

FREQUENCY_COLS = ["country", "agent"]

CIRCULAR_COLS = ["arrival_date_week_number", "booking_date_week_number"]