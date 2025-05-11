# 共通の修正ポイント
# - デバイス指定
# - データおよびモデルのデバイス移動
# - NumPy変換前のデバイス移動
# - パス指定の明確化
# - 特徴量の前処理強化（one-hotエンコード、スケーリング追加）
# - 対数変換の適用（log1p -> expm1）

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import jpholiday
import torch


def load_data(csv_path="data/spending_cleaned.csv") -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["date_time"])


def make_sequence_data(df: pd.DataFrame, seq_len: int = 14):
    df2 = df.copy()
    df2["date"] = df2["date_time"].dt.date
    ts = (
        df2
        .groupby("date", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "y"})
    )
    ts["date"] = pd.to_datetime(ts["date"])

    ts["dow"] = ts["date"].dt.weekday
    ts["is_holiday"] = ts["date"].apply(lambda d: 1.0 if jpholiday.is_holiday(d) else 0.0)
    ts["is_weekend"] = ts["dow"].apply(lambda x: 1.0 if x >= 5 else 0.0)

    def dist_from_payday(d):
        payday = d.replace(day=25)
        if d < payday:
            ref = payday
        else:
            ref = (payday + pd.offsets.MonthBegin(1))
        return float((d - ref).days)

    ts["days_from_payday"] = ts["date"].apply(dist_from_payday)

    scaler = StandardScaler()
    y_vals = ts["y"].values.reshape(-1, 1)
    y_scaled = scaler.fit_transform(y_vals).flatten()

    features = ["y_scaled", "dow", "is_holiday", "is_weekend", "days_from_payday"]
    ts["y_scaled"] = y_scaled

    X, Y = [], []
    for i in range(len(ts) - seq_len):
        block = ts.iloc[i:i + seq_len][features].values
        X.append(block)
        Y.append(y_scaled[i + seq_len])

    X_tensor = torch.tensor(np.stack(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1)

    return X_tensor, Y_tensor, scaler

def make_sequence_data_enhanced(df: pd.DataFrame, seq_len: int = 14):
    df2 = df.copy()
    df2["date"] = df2["date_time"].dt.date
    ts = (
        df2
        .groupby("date", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "y"})
    )
    ts["date"] = pd.to_datetime(ts["date"])
    ts = ts.sort_values("date").reset_index(drop=True)

    ts["y_prev_diff"] = ts["y"].diff().fillna(0)
    ts["y_roll7_mean"] = ts["y"].rolling(window=7, min_periods=1).mean()

    ts["dow"] = ts["date"].dt.weekday
    ts["is_holiday"] = ts["date"].apply(lambda d: 1.0 if d.weekday() >= 5 else 0.0)
    ts["is_weekend"] = ts["dow"].apply(lambda x: 1.0 if x >= 5 else 0.0)
    ts["is_month_start"] = ts["date"].dt.is_month_start.astype(float)
    ts["is_month_end"] = ts["date"].dt.is_month_end.astype(float)
    ts["month_sin"] = np.sin(2 * np.pi * ts["date"].dt.month / 12)
    ts["month_cos"] = np.cos(2 * np.pi * ts["date"].dt.month / 12)

    def dist_from_payday(d):
        payday = d.replace(day=25)
        if d < payday:
            ref = payday
        else:
            ref = (payday + pd.offsets.MonthBegin(1))
        return float((d - ref).days)

    ts["days_from_payday"] = ts["date"].apply(dist_from_payday)

    # 対数変換 -> スケーリング
    #ts["y_log"] = np.log1p(ts["y"])
    y_scaler = StandardScaler()
    #ts["y_scaled"] = y_scaler.fit_transform(ts[["y_log"]]).flatten()
    ts["y_scaled"] = y_scaler.fit_transform(ts[["y"]]).flatten()

    # One-hotエンコード（dow）を pandas で処理
    dow_encoded_df = pd.get_dummies(ts["dow"], prefix="dow")
    ts = pd.concat([ts.reset_index(drop=True), dow_encoded_df.reset_index(drop=True)], axis=1)

    # 他特徴量（連続値）のスケーリング
    cont_features = [
        "is_holiday", "is_weekend", "days_from_payday",
        "is_month_start", "is_month_end", "month_sin", "month_cos",
        "y_prev_diff", "y_roll7_mean"
    ]
    scaler_cont = StandardScaler()
    ts[cont_features] = scaler_cont.fit_transform(ts[cont_features])

    # 特徴量選択
    feature_cols = ["y_scaled"] + list(dow_encoded_df.columns) + cont_features
    X, Y = [], []
    for i in range(len(ts) - seq_len):
        block = ts.iloc[i:i + seq_len][feature_cols].values.astype(float)
        X.append(block)
        Y.append(ts.iloc[i + seq_len]["y_scaled"])

    X_tensor = torch.tensor(np.stack(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32).unsqueeze(1)
    date_seq = ts["date"].iloc[seq_len:].reset_index(drop=True)
    
    return X_tensor, Y_tensor, y_scaler, date_seq

