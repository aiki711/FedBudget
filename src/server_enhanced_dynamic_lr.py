# server_enhanced_dynamic_lr_fixed.py

import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters
import pandas as pd
from datetime import datetime

from train_pytorch_lstm import AttentionLSTMModel
from utils import load_data, make_sequence_data_enhanced
from config import SEQ_LEN, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

# ログ記録用リスト
log_records = []

# === ラウンドごとに動的に学習率を設定 ===
def dynamic_fit_config(round_number):
    base_lr = 0.01
    lr = base_lr * (0.5 ** (round_number - 1))
    print(f"📉 Round {round_number}: Setting client learning rate to {lr}")
    return {"lr": lr}

# === ログ保存用コールバック ===
def on_round_end(server_round, parameters, metrics, config):
    print(f"🔁 Round {server_round} ended")
    print(f"   - Metrics: {metrics}")
    log_records.append({
        "round": server_round,
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    })

# === 初期パラメータの作成 ===
def get_initial_parameters():
    # ダミーデータから特徴量数を取得
    df_raw = load_data()
    X, _, _, _ = make_sequence_data_enhanced(df_raw, SEQ_LEN)
    input_size = X.shape[2]

    model = AttentionLSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    weights = [val.cpu().detach().numpy() for val in model.parameters()]
    return ndarrays_to_parameters(weights)

def weighted_average(metrics):
    total_examples = sum(m["num_examples"] for m in metrics)
    avg_loss = sum(m["num_examples"] * m["loss"] for m in metrics) / total_examples
    return {"loss": avg_loss}

# === サーバ起動 ===
def main():
    print("🚀 Flower Server with FedAdam + dynamic LR starting...")

    strategy = fl.server.strategy.FedAdam(
        min_fit_clients=2,
        min_available_clients=2,
        eta=0.01,
        on_fit_config_fn=dynamic_fit_config,
        initial_parameters=get_initial_parameters(),
        #fit_metrics_aggregation_fn=weighted_average,
        #evaluate_metrics_aggregation_fn=weighted_average,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # ログ保存
    df_log = pd.DataFrame(log_records)
    df_log.to_csv("server_log_dynamic_lr.csv", index=False)
    print("📁 サーバーログを server_log_dynamic_lr.csv に保存しました")

if __name__ == "__main__":
    main()
