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


def make_ratio_sequence_data(df: pd.DataFrame, seq_len: int = 14):
    df["date"] = df["date_time"].dt.date
    df["date"] = pd.to_datetime(df["date"])
    df_grouped = df.groupby(["date", "category"])[["amount"]].sum().reset_index()
    df_pivot = df_grouped.pivot(index="date", columns="category", values="amount").fillna(0)
    df_pivot = df_pivot.sort_index()

    df_ratio = df_pivot.div(df_pivot.sum(axis=1), axis=0).fillna(0)
    df_ratio = df_ratio.replace([np.inf, -np.inf], 0)
    ratio_cols = df_ratio.columns.tolist()

    rolling_mean = df_pivot.rolling(window=3, min_periods=1).mean()
    diff = df_pivot.diff().fillna(0)
    std = df_pivot.rolling(window=3, min_periods=1).std().fillna(0)
    spike_flag = (np.abs(diff) > std * 2).astype(int)

    dow = df_ratio.index.dayofweek.values
    day = df_ratio.index.day.values
    month = df_ratio.index.month.values
    week = df_ratio.index.isocalendar().week.values
    is_start = (df_ratio.index.day <= 5).astype(int)
    is_end = (df_ratio.index.day >= 25).astype(int)

    features = pd.DataFrame({
        "day": day,
        "month": month,
        "week": week,
        "monthly_income": 300000,
        "is_payday": (day == 25).astype(int),
        "is_weekend": (dow >= 5).astype(int),
        "is_month_start": is_start,
        "is_month_end": is_end,
    }, index=df_ratio.index)
    dow_dummies = pd.get_dummies(dow, prefix="dow")
    features = pd.concat([features, dow_dummies, rolling_mean, spike_flag], axis=1).fillna(0)

    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(features)

    X, Y = [], []
    date_list = []
    for i in range(len(df_ratio) - seq_len):
        x_seq = X_scaled[i:i+seq_len]
        y_val = df_ratio.iloc[i + seq_len].values.astype(np.float32)
        X.append(x_seq)
        Y.append(y_val)
        date_list.append(df_ratio.index[i + seq_len])

    X_tensor = torch.tensor(np.stack(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.stack(Y), dtype=torch.float32)
    return X_tensor, Y_tensor, scaler_x, ratio_cols, date_list, df_ratio

def make_ratio_sequence_data_with_padding(df: pd.DataFrame, seq_len: int = 21):
    df["date"] = pd.to_datetime(df["date_time"]).dt.date
    df_grouped = df.groupby(["date", "category"])[["amount"]].sum().reset_index()
    df_pivot = df_grouped.pivot(index="date", columns="category", values="amount").fillna(0).sort_index()

    # 🟡 合計支出ゼロの日は除外
    df_pivot = df_pivot[df_pivot.sum(axis=1) > 0]

    df_ratio = df_pivot.div(df_pivot.sum(axis=1), axis=0).fillna(0)
    ratio_cols = df_ratio.columns.tolist()

    # 特徴量作成
    rolling_mean = df_pivot.rolling(window=3, min_periods=1).mean()
    diff = df_pivot.diff().fillna(0)
    std = df_pivot.rolling(window=3, min_periods=1).std().fillna(0)
    spike_flag = (np.abs(diff) > std * 2).astype(int)

    idx = pd.DatetimeIndex(df_ratio.index)
    dow = idx.dayofweek
    day = idx.day
    week = idx.isocalendar().week
    features = pd.DataFrame({
        "day": day,
        "week": week,
        "monthly_income": 300000,
        "is_payday": (day == 25).astype(int),
        "is_weekend": (dow >= 5).astype(int),
        "is_month_start": (day <= 5).astype(int),
        "is_month_end": (day >= 25).astype(int),
    }, index=idx)
    dow_dummies = pd.get_dummies(dow, prefix="dow")
    features = pd.concat([features, dow_dummies, rolling_mean, spike_flag], axis=1).fillna(0)

    # 🔧 スケーリング
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(features)

    # 🧩 パディング処理付きデータ生成
    X_tensor, Y_tensor, date_list = [], [], []
    for i in range(len(df_ratio)):
        start = max(0, i - seq_len)
        x_seq = X_scaled[start:i]
        y_val = df_ratio.iloc[i].values.astype(np.float32)

        # パディング（前に0を追加）
        if x_seq.shape[0] < seq_len:
            pad_len = seq_len - x_seq.shape[0]
            x_seq = np.vstack([np.zeros((pad_len, x_seq.shape[1])), x_seq])

        X_tensor.append(x_seq)
        Y_tensor.append(y_val)
        date_list.append(df_ratio.index[i])

    X_tensor = torch.tensor(np.stack(X_tensor), dtype=torch.float32)
    Y_tensor = torch.tensor(np.stack(Y_tensor), dtype=torch.float32)
    return X_tensor, Y_tensor, scaler_x, ratio_cols, date_list, df_ratio


