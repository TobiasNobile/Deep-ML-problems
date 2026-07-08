import pandas as pd

def solution(df):
    columns_to_drop = []
    for column in df.columns:
        missing_prop = df[column].isnull().sum()/len(df[column])
        if missing_prop > 0.5:
            columns_to_drop.append(column)
    df.drop(columns_to_drop, axis=1, inplace=True)

    count_row_na = df.isnull().sum(axis=1)
    df = df[count_row_na/len(df.columns) <= 0.5]

    nan_values = {c:None for c in df.columns}
    for c in nan_values:
        if pd.api.types.is_numeric_dtype(df[c]):
            nan_values[c] = df[c].mean()
        else:
            nan_values[c] = df[c].mode().iloc[0]

    return df.fillna(value=nan_values)
    