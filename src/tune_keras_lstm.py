# src/tune_keras_lstm.py

import numpy as np
import optuna
import mlflow
import mlflow.keras
from pathlib import Path

import tensorflow as tf
from mlflow_config import setup_mlflow
from utils import load_data, make_sequence_data, build_lstm_model
from config import DATA_CSV, SEQ_LEN, EXPERIMENT_TUNE


def objective(trial):
    # ハイパーパラ探索空間
    lstm_units       = trial.suggest_int("lstm_units", 32, 128, step=32)
    lstm_layers      = trial.suggest_int("lstm_layers", 1, 3)
    dense_units      = trial.suggest_int("dense_units", 16, 64, step=16)
    dense_layers     = trial.suggest_int("dense_layers", 1, 3)
    dropout          = trial.suggest_float("dropout", 0.0, 0.5)
    recurrent_dropout= trial.suggest_float("recurrent_dropout", 0.0, 0.5)
    lr           = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    batch_size   = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs       = 50

    # データ準備（multivariate + scaler を一度だけ適用）
    df_raw = load_data(str(DATA_CSV))
    X, y, scaler = make_sequence_data(df_raw, seq_len=SEQ_LEN)

    # ホールドアウト分割
    split = len(X) - SEQ_LEN
    X_train, y_train = X[:split], y[:split]
    X_test,  y_test  = X[split:], y[split:]

    with mlflow.start_run(nested=True):
        mlflow.log_params({
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "lr": lr,
            "dropout": dropout,
            "batch_size": batch_size,
            "seq_len": SEQ_LEN,
        })

        # モデル構築
        model = build_lstm_model(
            seq_len=SEQ_LEN,
            feature_dim=X.shape[2],
            lstm_units=lstm_units,
            lstm_layers=lstm_layers,
            dense_units=dense_units,
            dense_layers=dense_layers,             
            lr=lr,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,   
        )
        # 学習
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5, restore_best_weights=True
                )
            ],
            verbose=0,
        )

        # テストセットで予測 → 元スケールに戻して MAPE
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mlflow.log_metric("MAPE_holdout", mape)

    return mape

def main():
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_TUNE)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best params:", study.best_params)

if __name__ == "__main__":
    main()
