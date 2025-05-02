# src/utils.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import jpholiday
import tensorflow as tf


# ---------- Loader ----------
def load_data(csv_path="data/spending_cleaned.csv", *, prophet_format=False):
    df = pd.read_csv(csv_path, parse_dates=["date_time"])
    return df

def make_sequence_data(df: pd.DataFrame, seq_len: int = 14):
    """
    生データ df から以下を日次集計し、過去 seq_len 日分のシーケンスを作成:
      - 支出額 (amount)
      - 曜日 (dow: 0=Mon…6=Sun)
      - 祝日フラグ (is_holiday: 0/1)
      - 週末フラグ (is_weekend: 0/1)
      - 給与日からの距離 (days_from_payday)
    戻り値:
      X: ndarray (N, seq_len, 5)
      y: ndarray (N,)
      scaler: StandardScaler（支出のみのスケーラ）
    """
    # 1) 日次集計
    df2 = df.copy()
    df2["date"] = df2["date_time"].dt.date
    ts = (
        df2
        .groupby("date", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "y"})
    )
    ts["date"] = pd.to_datetime(ts["date"])
    # 2) 追加特徴量
    ts["dow"]         = ts["date"].dt.weekday
    ts["is_holiday"]  = ts["date"].apply(lambda d: 1.0 if jpholiday.is_holiday(d) else 0.0)
    ts["is_weekend"]  = ts["dow"].apply(lambda x: 1.0 if x >= 5 else 0.0)
    # 給料日距離: 毎月25日が給料日（例）として計算
    def dist_from_payday(d):
        payday = d.replace(day=25)
        if d < payday:
            ref = payday
        else:
            ref = (payday + pd.offsets.MonthBegin(1))
        return float((d - ref).days)
    ts["days_from_payday"] = ts["date"].apply(dist_from_payday)

    # 3) 支出を標準化
    scaler   = StandardScaler()
    y_vals   = ts["y"].values.reshape(-1,1)
    y_scaled = scaler.fit_transform(y_vals).flatten()

    # 4) シーケンス化
    features = ["y_scaled","dow","is_holiday","is_weekend","days_from_payday"]
    # まず y_scaled を列に戻す
    ts["y_scaled"] = y_scaled
    X, Y = [], []
    for i in range(len(ts) - seq_len):
        block = ts.iloc[i:i+seq_len][features].values    # shape (seq_len,5)
        X.append(block)
        Y.append(y_scaled[i + seq_len])
    X = np.stack(X, axis=0)   # (N, seq_len, 5)
    Y = np.array(Y)           # (N,)

    return X, Y, scaler

def build_lstm_model(
        seq_len: int,
        feature_dim: int,
        lstm_units: int = 64,
        lstm_layers: int = 2,
        dense_units: int = 32,
        dense_layers: int = 1,
        lr: float = 1e-3,
        dropout: float = 0.2,
        recurrent_dropout: float = 0.2,
        ) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(seq_len, feature_dim))
    x = inp

    # 多層・双方向 LSTM
    for i in range(lstm_layers):
        return_seq = (i < lstm_layers - 1)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                lstm_units,
                return_sequences=return_seq,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
            )
        )(x)
        # 各 LSTM 層の後にバッチ正規化
        x = tf.keras.layers.BatchNormalization()(x)

    # Dense 層＋ドロップアウト
    for _ in range(dense_layers):
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    out = tf.keras.layers.Dense(1, name="yhat")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=["mae"],
    )
    return model
