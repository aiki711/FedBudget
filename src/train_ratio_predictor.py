import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils import load_data, make_ratio_sequence_data
from config import DATA_CSV, SEQ_LEN
from budget_utils import (
    rebalance_budget,
    calc_daily_limits,
    simulate_with_user_budget,
    calculate_last_month_budget,
)

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

    print()

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

