import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
from utils import load_data, make_sequence_data_enhanced
from config import DATA_CSV, SEQ_LEN, EXPERIMENT_TUNE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
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
        attn_weights = torch.softmax(self.attn_linear(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output

def setup_mlflow():
    mlflow.set_tracking_uri("http://localhost:5000")    # ← 1 と同じホスト名/ポート
    mlflow.set_experiment("pytorch_baseline")           # スクリプトごとに実験名を変える

#自分の PC だけで使う	--host localhost で OK。外部ブラウザからはアクセス不可。
#チーム LAN 内で共有	--host 0.0.0.0 + 会社 LAN の IP でアクセス。
#                      必要なら Windows Defender / ufw で TCP 5000 を許可。
#クラウド (EC2/GCE)	    同上 + セキュリティグループでポート 5000 を開放。
#                      SSL 終端なら Nginx+Let'sEncrypt を前段に置く。

def objective(trial):
    lstm_units = trial.suggest_int("lstm_units", 32, 256, step=32)
    lstm_layers = trial.suggest_int("lstm_layers", 1, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    epochs = 50

    df_raw = load_data(str(DATA_CSV))
    X, y, scaler = make_sequence_data_enhanced(df_raw, seq_len=SEQ_LEN)

    split = len(X) - SEQ_LEN
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    X_train_tensor = X_train.clone().detach().float()
    y_train_tensor = y_train.clone().detach().float().reshape(-1, 1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_test_tensor = X_test.clone().detach().float()
    y_test_tensor = y_test.clone().detach().float().reshape(-1, 1)

    model = AttentionLSTMModel(X.shape[2], lstm_units, lstm_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "lstm_units": lstm_units,
            "lstm_layers": lstm_layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "seq_len": SEQ_LEN,
        })

        model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor.to(device)).cpu().numpy().flatten()

        y_pred = np.expm1(scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten())
        y_true = np.expm1(scaler.inverse_transform(y_test_tensor.cpu().numpy().reshape(-1, 1)).flatten())

        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mlflow.log_metric("MAPE_holdout", mape)

        # ✅ モデルをMLflowに保存
        model.cpu()
        mlflow.pytorch.log_model(model, artifact_path="model")

    return mape


def main():
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_TUNE)

    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=30)

    print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
