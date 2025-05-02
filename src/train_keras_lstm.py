# src/train_keras_lstm_scaled.py

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.keras
from mlflow_config import setup_mlflow
from utils import load_data, make_sequence_data, build_lstm_model
from config import DATA_CSV, MODELS_DIR, SEQ_LEN, EXPERIMENT_TRAIN

def main():
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_TRAIN)

    # データ読み込み
    df_raw = load_data(str(DATA_CSV))

    # シーケンス化 & スケーラ取得
    X_all, y_all, scaler = make_sequence_data(df_raw, SEQ_LEN)

    # ホールドアウト分割
    split = len(X_all) - SEQ_LEN
    X_train, y_train = X_all[:split], y_all[:split]
    X_test,  y_test  = X_all[split:], y_all[split:]

    with mlflow.start_run(run_name=EXPERIMENT_TRAIN):
        # モデル構築：config.SEQ_LEN, feature_dim は X_train.shape[2] から取得
        model = build_lstm_model(
            seq_len=SEQ_LEN,
            feature_dim=X_train.shape[2],
        )
        mlflow.log_param("seq_len", SEQ_LEN)

        # 学習
        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=50,
            batch_size=8,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0,
        )

        # 予測 → 逆スケーリング → MAPE
        y_pred_scaled = model.predict(X_test).flatten()
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mlflow.log_metric("MAPE_holdout", mape)

    # モデル保存
    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "keras_lstm_prophet_scaled.keras")
    print(f"✅ Scaled LSTM model saved, MAPE={mape:.2f}%")

if __name__ == "__main__":
    main()
