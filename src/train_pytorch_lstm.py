import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
import mlflow
from mlflow_config import setup_mlflow
from utils import load_data, make_sequence_data_enhanced
from config import DATA_CSV, MODELS_DIR, SEQ_LEN, BATCH_SIZE, LEARNING_RATE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, EPOCHS, EXPERIMENT_TRAIN
from pathlib import Path

# 固定ハイパーパラメータ
SEQ_LEN = 14
BATCH_SIZE = 8
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT = 0.0159
LEARNING_RATE = 0.00717663702416692
EPOCHS = 50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(AttentionLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attn_linear = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_linear(lstm_out), dim=1)  # shape: (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)              # shape: (batch, hidden_size * 2)
        output = self.fc(context)                                        # shape: (batch, 1)
        return output


def train(model, dataloader, val_data, criterion, optimizer, scheduler=None, epochs=EPOCHS, patience=5):
    model.train()
    val_X, val_y = val_data
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 検証損失計算
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X.to(device))
            val_loss = criterion(val_pred, val_y.to(device)).item()

        if scheduler:
            scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {epoch_loss/len(dataloader):.4f}, Val Loss = {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        model.train()


def evaluate(model, X_test, y_test, scaler):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy().flatten()
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1)).flatten()
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def main():
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_TRAIN)

    df_raw = load_data(str(DATA_CSV))
    X_all, y_all, scaler = make_sequence_data_enhanced(df_raw, SEQ_LEN)

    # スケーラーで逆変換
    y_true_values = scaler.inverse_transform(y_all.cpu().numpy().reshape(-1, 1)).flatten()

    # 日付列（SEQ_LEN分を除いた残り）を使ってDataFrame化
    dates = df_raw["date_time"].iloc[-len(y_all):].reset_index(drop=True)
    y_true_df = pd.DataFrame({"date_time": dates, "total": y_true_values})

    # 保存
    y_true_df.to_csv("y_true.csv", index=False)
    print("✅ 実測データを y_true.csv として保存しました。")

    split1 = int(len(X_all) * 0.7)
    split2 = int(len(X_all) * 0.85)

    X_train, y_train = X_all[:split1], y_all[:split1]
    X_val, y_val = X_all[split1:split2], y_all[split1:split2]
    X_test, y_test = X_all[split2:], y_all[split2:]

    X_train_tensor = X_train.clone().detach().float()
    y_train_tensor = y_train.clone().detach().float().reshape(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    X_val_tensor = X_val.clone().detach().float()
    y_val_tensor = y_val.clone().detach().float().reshape(-1, 1)

    X_test_tensor = X_test.clone().detach().float()
    y_test_tensor = y_test.clone().detach().float().reshape(-1, 1)

    model = AttentionLSTMModel(input_size=X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    with mlflow.start_run(run_name=EXPERIMENT_TRAIN):
        mlflow.log_param("seq_len", SEQ_LEN)
        train(model, train_loader, (X_val_tensor, y_val_tensor), criterion, optimizer, scheduler, epochs=EPOCHS, patience=5)
        mape = evaluate(model, X_test_tensor, y_test_tensor, scaler)
        mlflow.log_metric("MAPE_holdout", mape)
        mlflow.pytorch.log_model(model, artifact_path="models")

    global MODELS_DIR
    MODELS_DIR = Path(MODELS_DIR)
    MODELS_DIR.mkdir(exist_ok=True)
    model.cpu()
    mlflow.pytorch.log_model(model, artifact_path="model")
    torch.save(model.state_dict(), MODELS_DIR / "attention_lstm_scaled.pth")
    print(f"✅ Attention LSTM model saved, MAPE={mape:.2f}%")


if __name__ == "__main__":
    main()
