import pandas as pd
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical columns to numeric.
    Maps common binary categories, else applies label encoding.
    """
    for col in df.select_dtypes(include=['object']).columns:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals <= {'M', 'F'}:
            df[col] = df[col].map({'M': 1, 'F': 0})
        elif unique_vals <= {'Y', 'N'}:
            df[col] = df[col].map({'Y': 1, 'N': 0})
        elif unique_vals <= {'Yes', 'No'}:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
        else:
            df[col] = pd.factorize(df[col])[0]
    return df
