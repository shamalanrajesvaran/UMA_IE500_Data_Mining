from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale(df, method='standard'):
    # Scales the numerical (non-binary) columns of the DataFrame in-place using the specified method.
    
    # Select numerical columns
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Filter out binary columns (only values 0 and 1)
    non_binary_cols = [
        col for col in numeric_cols
        if not set(df[col].dropna().unique()).issubset({0, 1})
    ]

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaling method")

    df[non_binary_cols] = scaler.fit_transform(df[non_binary_cols])
