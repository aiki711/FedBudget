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

# 固定ハイパーパラメータ
SEQ_LEN = 14
BATCH_SIZE = 8
HIDDEN_SIZE = 256
NUM_LAYERS = 3
DROPOUT = 0.0159
LEARNING_RATE = 0.00717663702416692
EPOCHS = 50


# --- 給与日 (例として毎月25日) ---
PAYDAY_DAY = 25

# --- MLflow 実験名 ---
EXPERIMENT_TRAIN = "lstm_pytorch"
EXPERIMENT_TUNE  = "lstm_pytorch_tune"
EXPERIMENT_MULTI = "lstm_multi"
EXPERIMENT_MULTI_TUNE = "lstm_multi_tune"

# 必要なディレクトリをあらかじめ作成
for d in (MODELS_DIR, SAVED_MODELS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
