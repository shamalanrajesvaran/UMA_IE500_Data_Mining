from sklearn.model_selection import train_test_split
import pandas as pd

def split_df(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int | None = None,
    stratify: pd.Series | None = None,
    shuffle: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and test subsets.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to split.
    test_size : float, default=0.3
        Proportion of the data to include in the test split (0 < test_size < 1).
    random_state : int | None, default=None
        Random seed for reproducibility.
    stratify : pd.Series | None, default=None
        If given, splits are made so that the `stratify` target variable is
        evenly distributed in both splits.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
        shuffle=shuffle,
    )
    return train_df, test_df
