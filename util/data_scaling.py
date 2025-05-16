from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def scale(
    df: pd.DataFrame,
    method: str = "standard",
    columns: list[str] | None = None,
    *,
    return_scaler: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, object]:
    """
    Scale a DataFrame’s numeric columns with the chosen normalization technique.

    Parameters
    ----------
    df : pd.DataFrame
        Input data (left unmodified; a copy is returned).
    method : {'standard', 'minmax', 'robust'}, default='standard'
        • **standard** – zero-mean / unit-variance (`StandardScaler`)  
        • **minmax**  – rescale to [0, 1] (`MinMaxScaler`)  
        • **robust**  – median & IQR based (`RobustScaler`)
    columns : list[str] | None, default=None
        Columns to scale.  If *None*, all numeric columns are used.
    return_scaler : bool, keyword-only, default=False
        If *True*, also return the fitted scaler so you can reuse it
        (e.g. on validation / test data).

    Returns
    -------
    pd.DataFrame
        The scaled DataFrame (copy); or *(scaled_df, scaler)* if
        *return_scaler* is *True*.
    """
    # Map method → sklearn scaler class
    scaler_cls = {
        "standard": StandardScaler,
        "minmax":   MinMaxScaler,
        "robust":   RobustScaler,
    }.get(method.lower())

    if scaler_cls is None:
        raise ValueError("method must be 'standard', 'minmax', or 'robust'")

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    scaler = scaler_cls()
    scaled_df = df.copy()
    scaled_df[columns] = scaler.fit_transform(scaled_df[columns])

    return (scaled_df, scaler) if return_scaler else scaled_df
