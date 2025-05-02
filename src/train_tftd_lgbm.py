# src/train_tfdf_lgbm.py

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
from pathlib import Path

from mlflow_config import setup_mlflow
import mlflow
import mlflow.tensorflow

from utils import load_data, feature_engineer
from config import DATA_CSV, SAVED_MODELS_DIR, EXPERIMENT_TRAIN

def main():
    # MLflow 初期化
    setup_mlflow()
    mlflow.set_experiment(EXPERIMENT_TRAIN)

    # 1) データ読み込み＆前処理（LightGBM と同じ特徴量）
    df = load_data(str(DATA_CSV))
    df = feature_engineer(df)  # date, dow, month, lag1, lag7 を追加

    # 2) TF-DF 用 TensorFlow Dataset 化
    tf_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(
        df, label="amount", batch_size=len(df)
    )

    # 3) モデル定義 & 学習
    model = tfdf.keras.GradientBoostedTreesModel(
        num_trees=300,
        max_depth=6,
        name="tfdf_lgbm",
    )
    with mlflow.start_run(run_name="tfdf_category_baseline"):
        mlflow.log_param("num_trees", 300)
        mlflow.log_param("max_depth", 6)

        model.fit(tf_dataset)

        # MLflow に SavedModel としてログ
        mlflow.tensorflow.log_model(
            tf_saved_model_dir=str(SAVED_MODELS_DIR / "tfdf_lgbm_sm"),
            tf_meta_graph_tags=None,
            tf_signature_def_key=None,
            artifact_path="model"
        )
        print(f"✅ Logged TF-DF model to MLflow")

    # 4) ローカルにも SavedModel 形式で保存
    out_dir = SAVED_MODELS_DIR / "tfdf_lgbm_savedmodel"
    model.save(out_dir, include_optimizer=False)
    print(f"✅ TF-DF SavedModel exported → {out_dir}")

if __name__ == "__main__":
    main()
