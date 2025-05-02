# src/tune_tfdf_lgbm.py

import optuna
import numpy as np
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import tensorflow_decision_forests as tfdf

from mlflow_config import setup_mlflow
from utils import load_data, feature_engineer
from config import DATA_CSV, EXPERIMENT_TUNE

def objective(trial):
    # ── 探索するハイパーパラ ──
    num_trees    = trial.suggest_int("num_trees", 50, 500, step=50)
    max_depth    = trial.suggest_int("max_depth", 3, 12)
    subsample    = trial.suggest_float("subsample", 0.5, 1.0)
    learning_rate= trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
    # ────────────────────────

    # データ準備（LightGBMと同じ特徴量）
    df = load_data(str(DATA_CSV))
    df = feature_engineer(df)
    # TF-DF dataset 化
    # 80% を学習、20% をホールドアウトとして使う
    train_size = int(len(df) * 0.8)
    df_train, df_val = df.iloc[:train_size], df.iloc[train_size:]
    ds_train = tfdf.keras.pd_dataframe_to_tf_dataset(df_train, label="amount")
    ds_val   = tfdf.keras.pd_dataframe_to_tf_dataset(df_val,   label="amount")

    with mlflow.start_run(nested=True):
        # パラログ
        mlflow.log_params({
            "num_trees": num_trees,
            "max_depth": max_depth,
            "subsample": subsample,
            "learning_rate": learning_rate,
        })

        # モデル定義
        model = tfdf.keras.GradientBoostedTreesModel(
            num_trees=num_trees,
            max_depth=max_depth,
            subsample=subsample,
            learning_rate=learning_rate,
            name="tfdf_lgbm_tuned",
        )
        # 学習
        model.fit(ds_train, verbose=0)

        # バリデーションで予測
        val_ds_x = tfdf.keras.pd_dataframe_to_tf_dataset(df_val, label="amount", shuffle=False)
        preds = np.array([x["predictions"][0] for x in model.predict(val_ds_x)])
        y_true = df_val["amount"].values
        # MAPE
        mape = np.mean(np.abs((y_true - preds) / y_true)) * 100
        mlflow.log_metric("MAPE_val", mape)

    return mape

def main():
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_TUNE)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    print("Best params:", study.best_params)

if __name__ == "__main__":
    main()
