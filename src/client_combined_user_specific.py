# client_combined_user_specific.py

import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from argparse import ArgumentParser
from pathlib import Path

from train_pytorch_lstm import AttentionLSTMModel
from train_ratio_predictor import (
    TransformerRatioPredictor, 
    make_ratio_sequence_data, 
    rebalance_budget, 
    calc_daily_limits,
    simulate_with_user_budget,
    calculate_last_month_budget
    )
from utils import make_sequence_data_enhanced
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_user_data(user_id):
    user_file = Path("data/users") / f"{user_id}.csv"
    if not user_file.exists():
        raise FileNotFoundError(f"❌ User data not found: {user_file}")
    df_raw = pd.read_csv(user_file, parse_dates=["date_time"])
    return df_raw


def load_local_data(user_id):
    df_raw = load_user_data(user_id)

    # 総額モデル用
    X_total, y_total, scaler_total, _ = make_sequence_data_enhanced(df_raw, seq_len=SEQ_LEN)
    split = int(len(X_total) * 0.8)
    train_total = TensorDataset(X_total[:split], y_total[:split])
    test_total = TensorDataset(X_total[split:], y_total[split:])

    # 比率モデル用
    X_ratio, y_ratio, _, cat_cols, _, _ = make_ratio_sequence_data(df_raw, seq_len=SEQ_LEN)
    split_r = int(len(X_ratio) * 0.8)
    train_ratio = TensorDataset(X_ratio[:split_r], y_ratio[:split_r])
    test_ratio = TensorDataset(X_ratio[split_r:], y_ratio[split_r:])

    return (train_total, test_total, scaler_total, X_total), (train_ratio, test_ratio, cat_cols), X_ratio


def get_models(input_total, input_ratio, output_ratio):
    model_total = AttentionLSTMModel(input_size=input_total).to(device)
    model_ratio = TransformerRatioPredictor(input_size=input_ratio, embed_dim=64, num_heads=4, ff_hidden_dim=128, output_size=output_ratio).to(device)
    return model_total, model_ratio


def get_parameters(model_total, model_ratio):
    total_params = [val.cpu().detach().numpy() for val in model_total.parameters()]
    ratio_params = [val.cpu().detach().numpy() for val in model_ratio.parameters()]
    return total_params + ratio_params


def set_parameters(model_total, model_ratio, parameters):
    total_len = len(list(model_total.parameters()))
    for param, val in zip(model_total.parameters(), parameters[:total_len]):
        param.data = torch.tensor(val, dtype=param.data.dtype).to(device)
    for param, val in zip(model_ratio.parameters(), parameters[total_len:]):
        param.data = torch.tensor(val, dtype=param.data.dtype).to(device)


def train(model, dataset, loss_fn, epochs=1, lr=0.01, label=""):
    model.train()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"🧠 [{label}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")



def evaluate(model, dataset, loss_fn):
    model.eval()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    loss_total = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss_total += loss_fn(model(xb), yb).item()
    return loss_total / len(loader)


def predict_daily_spending(model, X_last, scaler, days=7):
    model.eval()
    preds = []
    current_input = X_last.clone().detach()
    for _ in range(days):
        with torch.no_grad():
            y_scaled_pred = model(current_input).cpu().numpy()[0][0]
            y_pred = scaler.inverse_transform([[y_scaled_pred]])[0][0]
            preds.append(y_pred)
            next_input = current_input[0].clone().numpy()
            next_input[:-1] = next_input[1:]
            next_input[-1][0] = y_scaled_pred
            current_input = torch.tensor([next_input], dtype=torch.float32)
    return preds


def post_inference(user_id, model_total, model_ratio, X_total, scaler, X_ratio, cat_cols):
    today = datetime.today().date()
    start_date = pd.to_datetime(today)
    last_day = start_date.replace(day=1) + pd.offsets.MonthEnd(0)
    n_days = max((last_day - start_date).days, 7)

    model_total.eval()
    preds = []
    dates = []
    current_input = X_total[-1:].clone().detach()
    seventh_day_pred = None

    for i in range(n_days):
        with torch.no_grad():
            y_scaled_pred = model_total(current_input).cpu().numpy()[0][0]
            y_pred = scaler.inverse_transform([[y_scaled_pred]])[0][0]
            preds.append(y_pred)
            dates.append(today + timedelta(days=i + 1))
            if i == 6:
                seventh_day_pred = y_pred

            next_input = current_input[0].clone().numpy()
            next_input[:-1] = next_input[1:]
            next_input[-1][0] = y_scaled_pred
            current_input = torch.tensor([next_input], dtype=torch.float32)

    df_pred = pd.DataFrame({"date_time": dates, "predicted_total": preds})
    df_pred.to_csv(f"predicted_total_daily_{user_id}.csv", index=False)

    with torch.no_grad():
        ratio_pred = model_ratio(X_ratio[-1:].to(device)).cpu().numpy()[0]

    total_month_pred = sum(preds)
    forecast_cat_month = {cat: ratio_pred[i] * total_month_pred for i, cat in enumerate(cat_cols)}
    pd.DataFrame.from_dict(forecast_cat_month, orient="index", columns=["予測額"]).to_csv(f"forecast_category_{user_id}.csv")

    if seventh_day_pred is not None:
        forecast_cat_7day = {cat: ratio_pred[i] * seventh_day_pred for i, cat in enumerate(cat_cols)}
        pd.DataFrame.from_dict(forecast_cat_7day, orient="index", columns=["予測額"]).to_csv(f"forecast_category_7day_{user_id}.csv")

    custom_budget = {
        "total_budget": 200000,
        "food": 25000,
        "transport": 12000,
        "entertainment": 10000,
        "clothing_beauty_daily": 15000,
        "utilities": 20000,
        "social": 15000,
        "other": 8000,
    }

    rebalance_df = rebalance_budget(forecast_cat_month, custom_budget)
    rebalance_df.to_csv(f"rebalance_proposal_{user_id}.csv", index=False)

    daily_limit_df = calc_daily_limits(forecast_cat_month)
    daily_limit_df.to_csv(f"daily_limits_{user_id}.csv", index=False)

    print(f"📁 ユーザー {user_id} の月末・7日後予測、予算提案を保存しました")

    simulate_with_user_budget(model_ratio, X_ratio[-10:], cat_cols, custom_budget)

    # ✅ 先月の支出と比較したリスク評価
    df_raw = load_user_data(user_id)
    target_month = (datetime.today().replace(day=1) - pd.DateOffset(months=1)).strftime("%Y-%m")
    last_month_budget = calculate_last_month_budget(df_raw, target_month)
    print(f"\n🧮 先月（{target_month}）の支出ベースのリスク評価")
    simulate_with_user_budget(model_ratio, X_ratio[-10:], cat_cols, last_month_budget)


class FLCombinedClient(fl.client.NumPyClient):
    def __init__(self, user_id):
        (self.train_total, self.test_total, self.scaler_total, self.X_total),         (self.train_ratio, self.test_ratio, self.cat_cols),         self.X_ratio = load_local_data(user_id)
        input_total = self.X_total.shape[2]
        input_ratio = self.X_ratio.shape[2]
        output_ratio = self.train_ratio.tensors[1].shape[1]
        self.model_total, self.model_ratio = get_models(input_total, input_ratio, output_ratio)
        self.user_id = user_id

    def get_parameters(self, config):
        return get_parameters(self.model_total, self.model_ratio)

    def fit(self, parameters, config):
        set_parameters(self.model_total, self.model_ratio, parameters)
        lr = config.get("lr", LEARNING_RATE)
        train(self.model_total, self.train_total, nn.MSELoss(), lr=lr, label="model_total")
        train(self.model_ratio, self.train_ratio, nn.MSELoss(), lr=lr, label="model_ratio")

        # ✅ モデル保存処理
        torch.save(self.model_total.state_dict(), f"model_total_{self.user_id}.pt")
        torch.save(self.model_ratio.state_dict(), f"model_ratio_{self.user_id}.pt")
        print(f"💾 モデルを保存しました: model_total_{self.user_id}.pt, model_ratio_{self.user_id}.pt")

        return get_parameters(self.model_total, self.model_ratio), len(self.train_total), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model_total, self.model_ratio, parameters)
        loss_total = evaluate(self.model_total, self.test_total, nn.MSELoss())
        loss_ratio = evaluate(self.model_ratio, self.test_ratio, nn.MSELoss())
        return float(loss_total + loss_ratio), len(self.test_total), {}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--user_id", type=str, required=True, help="ユーザーID (例: U001)")
    args = parser.parse_args()

    client = FLCombinedClient(args.user_id)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    print(f"✅ FL完了。ユーザー {args.user_id} の推論と提案を生成中...")
    post_inference(args.user_id, client.model_total, client.model_ratio, client.X_total, client.scaler_total, client.X_ratio, client.cat_cols)
