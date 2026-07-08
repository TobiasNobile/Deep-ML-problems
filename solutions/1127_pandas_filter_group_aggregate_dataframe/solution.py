import pandas as pd

def solution(df):
    df = df[df["status"] == "completed"]
    total = df.groupby("region", as_index=False)["amount"].sum()
    return total.sort_values(by="amount", ascending=False).head(10)