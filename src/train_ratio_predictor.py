import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils import load_data
from config import DATA_CSV, SEQ_LEN
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerRatioPredictor(nn.Module):
    def __init__(self, input_size, embed_dim, num_heads, ff_hidden_dim, output_size):
        super().__init__()
        self.input_proj = nn.Linear(input_size, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_hidden_dim)
        self.fc = nn.Linear(embed_dim, output_size)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return F.softmax(self.fc(x), dim=1)


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


def rebalance_budget(predicted, budget_dict):
    df = pd.DataFrame({"カテゴリ": list(predicted.keys()), "予測額": list(predicted.values())})
    df["予算"] = df["カテゴリ"].map(budget_dict)
    df["残予算"] = df["予算"] - df["予測額"]
    df["不足額"] = df["予測額"] - df["予算"]
    df["不足額"] = df["不足額"].apply(lambda x: x if x > 0 else 0)
    df["余剰可能額"] = df["残予算"].apply(lambda x: x if x > 0 else 0)

    shortfall_df = df[df["不足額"] > 0].copy()
    surplus_df = df[df["余剰可能額"] > 0].copy()

    total_shortfall = shortfall_df["不足額"].sum()
    total_surplus = surplus_df["余剰可能額"].sum()

    if total_surplus > 0:
        shortfall_df["補填額"] = shortfall_df["不足額"].apply(
            lambda x: min(x, total_surplus * (x / total_shortfall))
        )
    else:
        shortfall_df["補填額"] = 0

    return shortfall_df[["カテゴリ", "予算", "予測額", "補填額"]]


def calc_daily_limits(budget_dict):
    today = datetime.today()
    end_of_month = today.replace(day=1) + pd.offsets.MonthEnd(1)
    remaining_days = (end_of_month - today).days

    category_limits = {
        cat: {
            "残日数支出上限": budget_dict[cat],
            "1日あたり上限": budget_dict[cat] / remaining_days if remaining_days > 0 else 0
        } for cat in budget_dict
    }

    return pd.DataFrame.from_dict(category_limits, orient="index").reset_index().rename(columns={"index": "カテゴリ"})


def train_ratio_model():
    df_raw = load_data(str(DATA_CSV))
    X, Y, scaler, cat_cols, dates, df_ratio = make_ratio_sequence_data(df_raw)

    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], Y[:split]
    X_test, y_test = X[split:], Y[split:]
    date_test = dates[split:]
    ratio_test = df_ratio.iloc[split + SEQ_LEN:]

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = TransformerRatioPredictor(input_size=X.shape[2], embed_dim=64, num_heads=4, ff_hidden_dim=128, output_size=Y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(20):
        model.train()
        epoch_loss = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(loader):.4f}")

    return model, X_test, y_test, cat_cols, date_test, df_raw, ratio_test


def plot_ratio_forecast(model, X_test, cat_cols, dates, total_budget):
    model.eval()
    with torch.no_grad():
        pred_ratios = model(X_test).numpy()

    pred_amounts = pred_ratios * total_budget
    df_pred = pd.DataFrame(pred_amounts, columns=cat_cols, index=dates)

    df_pred.plot(figsize=(12, 6), marker='o')
    plt.title("カテゴリ別 支出予測の推移")
    plt.ylabel("予測金額 (円)")
    plt.xlabel("日付")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ratio_comparison(pred_ratios, actual_ratios, cat_cols, dates):
    df_pred = pd.DataFrame(pred_ratios, columns=cat_cols, index=dates)
    df_true = actual_ratios.iloc[:len(df_pred)]

    fig, axes = plt.subplots(nrows=len(cat_cols), ncols=1, figsize=(12, 2.5 * len(cat_cols)), sharex=True)
    for i, cat in enumerate(cat_cols):
        axes[i].plot(df_true.index, df_true[cat], label="実測", marker="o")
        axes[i].plot(df_pred.index, df_pred[cat], label="予測", marker="x")
        axes[i].set_title(cat)
        axes[i].legend()
        axes[i].grid(True)
    plt.tight_layout()
    plt.show()


def simulate_with_user_budget(model, X_test, cat_cols, user_budget_dict):
    model.eval()
    with torch.no_grad():
        pred_ratios = model(X_test).numpy()

    avg_ratios = pred_ratios.mean(axis=0)
    risk_report = []

    for i, cat in enumerate(cat_cols):
        predicted_ratio = avg_ratios[i]
        predicted_amount = predicted_ratio * user_budget_dict.get("total_budget", 100000)
        budget = user_budget_dict.get(cat, predicted_amount)

        if predicted_amount > budget * 1.2:
            risk = "🔴 高リスク"
        elif predicted_amount > budget:
            risk = "🟠 中リスク"
        else:
            risk = "🟢 低リスク"

        risk_report.append((cat, predicted_amount, budget, risk))

    print("\n🧮 シミュレーション結果：予算に基づくリスク評価")
    print("カテゴリ\t予測額\t予算\tリスク")
    for cat, pred_amt, budget, risk in risk_report:
        print(f"{cat}\t{pred_amt:,.0f}\t{budget:,.0f}\t{risk}")


def calculate_last_month_budget(df, target_month):
    # 1. 文字列から datetime に変換（強制的に）
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")

    # 2. タイムゾーンがある場合のみ tz_localize(None)
    if df["date_time"].dt.tz is not None:
        df["date_time"] = df["date_time"].dt.tz_localize(None)

    # 3. 月抽出（文字列化）
    df["month"] = df["date_time"].dt.to_period("M").astype(str)

    # 4. 月フィルタ
    last_month_df = df[df["month"] == target_month]

    if last_month_df.empty:
        print(f"⚠️ {target_month} のデータが見つかりません。すべて0として扱います。")
        return {
            "total_budget": 0,
            "food": 0,
            "transport": 0,
            "entertainment": 0,
            "clothing_beauty_daily": 0,
            "utilities": 0,
            "social": 0,
            "other": 0,
        }

    result = last_month_df.groupby("category")["amount"].sum().to_dict()
    total = sum(result.values())
    result = {k: float(v) for k, v in result.items()}
    result["total_budget"] = total
    return result


if __name__ == "__main__":
    model, X_test, y_test, cat_cols, dates, df_raw, ratio_test = train_ratio_model()

    total_budget = 100000
    plot_ratio_forecast(model, X_test, cat_cols, dates, total_budget)

    with torch.no_grad():
        pred_ratios = model(X_test).numpy()
    plot_ratio_comparison(pred_ratios, ratio_test, cat_cols, dates)

    custom_budget = {
        "total_budget": 120000,
        "food": 25000,
        "transport": 12000,
        "entertainment": 10000,
        "clothing_beauty_daily": 15000,
        "utilities": 20000,
        "social": 15000,
        "other": 8000,
    }
    simulate_with_user_budget(model, X_test, cat_cols, custom_budget)

    budget_from_history = calculate_last_month_budget(df_raw, "2025-04")
    simulate_with_user_budget(model, X_test, cat_cols, budget_from_history)

        # 📊 予算内リバランス提案の出力
    print("\n📊 予算内リバランス提案（赤字カテゴリへの補填額）")
    rebalance_df = rebalance_budget({cat: pred_ratios.mean(axis=0)[i] * custom_budget["total_budget"] for i, cat in enumerate(cat_cols)}, custom_budget)
    print(rebalance_df.to_string(index=False))

    # 📆 月末までの残り日数と日割り支出上限の出力
    print("\n📆 月末までの残り日数と日割り支出上限（等配分ベース）")
    daily_limit_df = calc_daily_limits(custom_budget)
    print(daily_limit_df.to_string(index=False))

