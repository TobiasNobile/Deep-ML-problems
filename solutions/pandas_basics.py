import pandas as pd

def filter_group_aggregate(df):
    df = df[df["status"] == "completed"]
    total = df.groupby("region", as_index=False)["amount"].sum()
    return total.sort_values(by="amount", ascending=False).head(10)

def handle_missing_data(df):
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

def merge(df1, df2, df3):
    left = pd.merge(df1, df2, on='emp_id', how='inner')
    return pd.merge(left, df3, on='emp_id', how='left')

def dedup_standardize_impute(df):
    df['name'] = df['name'].str.strip().str.title()
    df['date'] = pd.to_datetime(df['date'].str.strip()).dt.strftime('%Y-%m-%d')
    df = df.drop_duplicates().reset_index(drop=True)
    df['value'] = df['value'].fillna(df['value'].mean())
    return df
