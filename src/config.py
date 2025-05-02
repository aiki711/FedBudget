# src/config.py

from pathlib import Path

# --- プロジェクトディレクトリパス ---
# このファイル(config.py)を起点として project/ フォルダを指す
BASE_DIR = Path(__file__).resolve().parents[1]

# --- データパス・出力ディレクトリ ---
DATA_CSV         = BASE_DIR / "data" / "spending_cleaned.csv"
MODELS_DIR       = BASE_DIR / "models"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
OUTPUT_DIR       = BASE_DIR / "output"

# --- LSTM シーケンス長 ---
SEQ_LEN = 14

# --- 給与日 (例として毎月25日) ---
PAYDAY_DAY = 25

# --- MLflow 実験名 ---
EXPERIMENT_TRAIN = "keras_lstm_prophet_scaled"
EXPERIMENT_TUNE  = "keras_lstm_optuna"

# 必要なディレクトリをあらかじめ作成
for d in (MODELS_DIR, SAVED_MODELS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
