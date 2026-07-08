import pandas as pd

def solution(df):
    df = df[df["status"] == "completed"]
    total = df.groupby("region", as_index=False)["amount"].sum()
    return total.sort_values(by="amount", ascending=False).head(10)

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

def solution(df1, df2, df3):
    left = pd.merge(df1, df2, on='emp_id', how='inner')
    return pd.merge(left, df3, on='emp_id', how='left')
