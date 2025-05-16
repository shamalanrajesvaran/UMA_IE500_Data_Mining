import pandas as pd
from util.feature_engineering import feature_engineering
from util.encode import encode, ONEHOT_COLS, FREQUENCY_COLS, CIRCULAR_COLS
from util.split import split_df
from util.data_cleaning import data_cleaning
from util.handle_outlier import handle_outlier
from util.data_scaling import scale

# file import
file_dir = "/Users/bofanchen/Desktop/data_mining/hotel_bookings.csv"
df = pd.read_csv(file_dir)

# generate additional features and delete unusable features
df = feature_engineering(df)

# apply encoding
df = encode(df,method="onehot", columns=ONEHOT_COLS)
df = encode(df,method="frequency", columns=FREQUENCY_COLS)
df = encode(df,method="circular", columns=CIRCULAR_COLS)

# splitting the data
train_data, temp_test_data = split_df(df, test_size=0.30, random_state=42)
val_data, holdout_test_data = split_df(temp_test_data, test_size=0.50, random_state=42)

# drop unusable rows
train_data = data_cleaning(train_df=train_data, df=train_data)
val_data = data_cleaning(train_df=train_data, df=val_data)
holdout_test_data = data_cleaning(train_df=train_data, df=holdout_test_data)

# handle outliers in the training set
train_data = handle_outlier(train_data)

# apply scaling to the data
train_data = scale(train_data, method="minmax")
val_data = scale(val_data, method="minmax")
holdout_test_data = scale(holdout_test_data, method="minmax")