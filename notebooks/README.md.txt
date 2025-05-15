# 家計簿予測連合学習システム

## 📦 プロジェクト概要
本プロジェクトは、ユーザーごとの家計支出データをもとに、
- 日次支出合計の予測（LSTM + Attention）
- カテゴリ別支出比率の予測（Transformer）
を行い、個人に最適化されたリスク評価と予算提案を実施します。

さらに、Flower を用いた Federated Learning に対応しており、
ユーザー端末上で分散学習を行いながらプライバシーを保護します。

---

## 📁 ディレクトリ構成
```
project/
├── config.py
├── mlflow_config.py
├── train_pytorch_lstm.py
├── train_ratio_predictor.py
├── export_to_onnx.py
├── client_combined_user_specific.py
├── server_enhanced_dynamic_lr.py
├── utils.py
├── budget_utils.py
├── models/                 # モデル保存先
├── data/
│   └── users/              # ユーザー別CSV格納
├── output/                # 推論CSV出力先
└── saved_models/          # 任意保存先
```

---

## ✅ セットアップ手順

### 0.ディレクトリの場所
projectで実行する

### 1. 仮想環境構築
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. 必要パッケージのインストール
```bash
pip install -r requirements.txt
```

---

## 🧪 学習手順

### 1. 合計支出モデルの学習
```bash
python src/train_pytorch_lstm.py
```

### 2. カテゴリ比率モデルの学習
```bash
python src/train_ratio_predictor.py
```

---

## 🔁 連合学習の実行

### サーバー起動（別ターミナル）
```bash
python server_enhanced_dynamic_lr.py
```

### クライアント起動
```bash
python client_combined_user_specific.py --user_id U001
```

---

## 📤 モデル変換
```bash
# LSTMモデル（合計支出）をONNXへ
python export_to_onnx.py --mode total --pt_path models/attention_lstm_scaled.pth --onnx_path models/attention_lstm_scaled.onnx --input_size 17

# Transformerモデル（カテゴリ比率）をONNXへ
python export_to_onnx.py --mode ratio --pt_path models/ratio_model.pth --onnx_path models/ratio_model.onnx --input_size 29
```

---

## 📈 出力されるCSV一覧
- `predicted_total_daily_*.csv`：日次支出合計予測
- `forecast_category_*.csv`：月末カテゴリ別支出予測
- `forecast_category_7day_*.csv`：7日後カテゴリ別予測
- `rebalance_proposal_*.csv`：赤字カテゴリへの補填提案
- `daily_limits_*.csv`：カテゴリ別日割り支出上限

---

## 📚 requirements.txt
```
torch
numpy
pandas
scikit-learn
matplotlib
mlflow
flwr
jpholiday
```

---

## 📝 備考
- Flower サーバとクライアントは別ターミナルで起動してください。
- MLflow ログは `mlruns/` に保存されます。
- データは `data/users/Uxxx.csv` に配置することでユーザー別に実験が可能です。
