import pandas as pd

def data_cleaning(train_df: pd.DataFrame,
                  df: pd.DataFrame
                  ) -> dict[str, int | float | str]:
    """
    Clean *train_df* and *other_df* in place.

    Actions (all done **in place**)
    -------------------------------
    1. Compute the training-set modes for:
         • children  
         • total_people  
         • is_solo_traveler
    2. Fill NaNs in those three columns for **both** dataframes with the
       training-set modes.
    3. Drop rows where  (adults == 0)  &  (babies > 0)  in **both** dataframes
       (i.e., keep rows that have at least one adult *or* have no babies).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split used to derive the fill values.
    other_df : pd.DataFrame
        Validation / test / hold-out split to clean with the same rules.

    Returns
    -------
    mode_values : dict
        {"children": <mode>, "total_people": <mode>, "is_solo_traveler": <mode>}
        (useful if you need to clean more datasets later).
    """
    # 1. learn modes from the training data
    mode_values = {
        "children":         train_df["children"].mode()[0],
        "total_people":     train_df["total_people"].mode()[0],
        "is_solo_traveler": train_df["is_solo_traveler"].mode()[0],
    }

    # 2. helper to apply fills and row drops on any dataframe (in place)
    def _apply(df: pd.DataFrame) -> None:
        df["children"].fillna(        mode_values["children"],        inplace=True)
        df["total_people"].fillna(    mode_values["total_people"],    inplace=True)
        df["is_solo_traveler"].fillna(mode_values["is_solo_traveler"], inplace=True)

        # drop rows that violate the adults/babies consistency rule
        bad_rows = df[(df["adults"] == 0) & (df["babies"] > 0)].index
        df.drop(index=bad_rows, inplace=True)

    _apply(train_df)
    _apply(df)

    return df
