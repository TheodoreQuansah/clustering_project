import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer


def train_val_test(df):
    """
    Split the DataFrame into training, validation, and test sets.

    Returns:
    pd.DataFrame: Training, validation, and test DataFrames.
    """
    seed = 42
    train, val_test = train_test_split(df, train_size=0.7, random_state=seed)
    val, test = train_test_split(val_test, train_size=0.5, random_state=seed)
    
    # Return the three datasets
    return train, val, test


def get_dummies(train, val, test):
    """
    Convert specified columns into dummies and rename them.

    Parameters:
    train (pd.DataFrame): Training dataset.
    val (pd.DataFrame): Validation dataset.
    test (pd.DataFrame): Test dataset.

    Returns:
    train, val, test (pd.DataFrames): Modified datasets with dummies and renamed columns.
    """
    # Specify columns to convert into dummies
    columns_to_convert = ['wine_type']

    # Perform conversion
    train = pd.get_dummies(train, columns=columns_to_convert)
    val = pd.get_dummies(val, columns=columns_to_convert)
    test = pd.get_dummies(test, columns=columns_to_convert)

    return train, val, test


def xy_split(df):
    """
    Split a DataFrame into features (X) and the target variable (y) by dropping the 'tax_value' column.

    Parameters:
    df (pd.DataFrame): DataFrame to be split.

    Returns:
    X (pd.DataFrame): Features.
    y (pd.Series): Target variable.
    """
    return df.drop(columns=['quality']), df.quality



def scaled_data(train, val, test, scaler_type='standard'):
    """
    Scale numerical features in train, val, and test datasets using various scaling techniques.

    Parameters:
    train (pd.DataFrame): Training dataset.
    val (pd.DataFrame): Validation dataset.
    test (pd.DataFrame): Test dataset.
    scaler_type (str): Type of scaler to use ('standard', 'minmax', 'robust', 'quantile').

    Returns:
    train, val, test (pd.DataFrames): Modified datasets with scaled numerical features.
    """
    # Initialize the selected scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    else:
        raise ValueError("Invalid scaler_type. Choose from 'standard', 'minmax', 'robust', 'quantile'.")

    # Exclude 'cluster' and 'quality' from features to scale
    features_to_scale = [col for col in train.columns if col not in ['cluster', 'quality']]
    
    # Fit the scaler on the training data and transform all sets
    train[features_to_scale] = scaler.fit_transform(train[features_to_scale])
    val[features_to_scale] = scaler.transform(val[features_to_scale])
    test[features_to_scale] = scaler.transform(test[features_to_scale])

    return train, val, test
