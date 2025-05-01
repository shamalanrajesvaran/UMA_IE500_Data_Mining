#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#SMOTE oversampling function

from imblearn.over_sampling import SMOTE

def apply_smote(X, y, sampling_strategy='auto', k_neighbors=5, random_state=42, n_jobs=None):
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to balance class distribution.

    Parameters:
    ------------
    X : array-like or DataFrame
        Feature matrix.

    y : array-like or Series
        Target vector.

    sampling_strategy : str, float, dict or callable (default='auto')
        Defines the sampling strategy. Common options:
        - 'auto': resample minority class to match majority class
        - float: resample minority class to achieve desired ratio
        - dict: specify number of samples per class
        - callable: custom sampling logic

    k_neighbors : int (default=5)
        Number of nearest neighbors to use when generating synthetic samples.

    random_state : int (default=42)
        Controls randomness for reproducibility.

    n_jobs : int or None (default=None)
        Number of CPU cores to use during resampling. Use -1 for all available cores.

    Returns:
    --------
    X_resampled : array-like
        Resampled feature matrix.

    y_resampled : array-like
        Resampled target vector.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
        n_jobs=n_jobs
    )
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

