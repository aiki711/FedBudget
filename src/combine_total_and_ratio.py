import torch
import pandas as pd
from datetime import timedelta, datetime
from train_ratio_predictor import (
    train_ratio_model,
    simulate_with_user_budget,
    rebalance_budget,
    calc_daily_limits
)
from train_pytorch_lstm import (
    make_sequence_data_enhanced, 
    AttentionLSTMModel
)

from utils import load_data
from config import DATA_CSV, SEQ_LEN, MODELS_DIR, HIDDEN_SIZE, NUM_LAYERS, DROPOUT

# ==== 1. モデルパス（固定モデルを使用） ====
model_path = MODELS_DIR / "attention_lstm_scaled.pth"

# ==== データ読み込みと準備 ====
df_raw = load_data(str(DATA_CSV))
X_total, y_total, scaler, date_seq = make_sequence_data_enhanced(df_raw, SEQ_LEN)
X_pred = X_total[-1:].to("cpu")  # 🔧 ここを忘れずに！
today = datetime.today().date()
start_date = pd.to_datetime(today)

# ==== 3. モデル構築と読み込み（固定パラメータ） ====
input_size = X_total.shape[2]
model = AttentionLSTMModel(
    input_size=input_size,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()
total_model = model

# ==== 日次予測：月末まで1日ずつ予測 ====
last_day = start_date.replace(day=1) + pd.offsets.MonthEnd(0)
n_days = max((last_day - start_date).days, 7)

preds = []
dates = []
seventh_day_pred = None

for i in range(n_days):
    with torch.no_grad():
        y_pred = model(X_pred).cpu().numpy()[0][0]
        y_pred_inv = scaler.inverse_transform([[y_pred]])[0][0]
        preds.append(y_pred_inv)
        dates.append(start_date + timedelta(days=i + 1))
        if i == 6:
            seventh_day_pred = y_pred_inv

    # 次の入力を構築
    next_input = X_pred[0].clone().numpy()
    next_input[:-1] = next_input[1:]
    next_input[-1] = y_pred
    X_pred = torch.tensor([next_input], dtype=torch.float32)

# ==== 日次予測結果を保存 ====
df_pred = pd.DataFrame({"date_time": dates, "predicted_total": preds})
df_pred.to_csv("predicted_total_daily.csv", index=False)
print("📁 日次予測を predicted_total_daily.csv に保存しました")

# ==== 月末のカテゴリ別予測 ====
ratio_model, X_ratio, y_ratio, cat_cols, _, _, _ = train_ratio_model()
X_ratio_last = X_ratio[-1:].to("cpu")
ratio_pred = ratio_model(X_ratio_last).detach().numpy()[0]
total_month_pred = sum(preds)

# === 予測された合計支出（月末と7日後） ===
total_month_pred = sum(preds)
seventh_day_total = seventh_day_pred

# === ratio モデルの取得と予測 ===
ratio_model, X_ratio, _, cat_cols, _, _, _ = train_ratio_model()
X_ratio_last = X_ratio[-1:].to("cpu")
ratio_model.eval()
with torch.no_grad():
    ratio_pred = ratio_model(X_ratio_last).numpy()[0]

# === カテゴリ別予測（月末） ===
forecast_cat_month = {
    cat: ratio_pred[i] * total_month_pred for i, cat in enumerate(cat_cols)
}
pd.DataFrame.from_dict(forecast_cat_month, orient="index", columns=["予測額"]).to_csv("forecast_category.csv")

# === カテゴリ別予測（7日後） ===
if seventh_day_pred is not None:
    forecast_cat_7day = {
        cat: ratio_pred[i] * seventh_day_total for i, cat in enumerate(cat_cols)
    }
    pd.DataFrame.from_dict(forecast_cat_7day, orient="index", columns=["予測額"]).to_csv("forecast_category_7day.csv")

# === 予算リスク評価 + 日割り上限 + リバランス ===
custom_budget = {
    "total_budget": 200000,  # ユーザー指定 or デフォルト値
    "food": 25000,
    "transport": 12000,
    "entertainment": 10000,
    "clothing_beauty_daily": 15000,
    "utilities": 20000,
    "social": 15000,
    "other": 8000,
}
simulate_with_user_budget(ratio_model, X_ratio[-10:], cat_cols, custom_budget)

# === リバランス提案の出力と保存 ===
rebalance_df = rebalance_budget(forecast_cat_month, custom_budget)
print("\n📊 リバランス提案（カテゴリ別）：")
print(rebalance_df)
rebalance_df.to_csv("rebalance_proposal.csv", index=False)
print("📁 リバランス提案を rebalance_proposal.csv に保存しました")

# === 1日あたりの支出上限の出力と保存 ===
daily_limit_df = calc_daily_limits(custom_budget)
print("\n📆 1日あたりの支出上限：")
print(daily_limit_df)
daily_limit_df.to_csv("daily_limits.csv", index=False)
print("📁 1日あたりの支出上限を daily_limits.csv に保存しました")


forecast_cat_month = {
    cat: ratio_pred[i] * total_month_pred for i, cat in enumerate(cat_cols)
}
pd.DataFrame.from_dict(forecast_cat_month, orient="index", columns=["予測額"]).to_csv("forecast_category.csv")
print("📁 月末カテゴリ別予測を forecast_category.csv に保存しました")

# ==== 7日後のカテゴリ別予測 ====
if seventh_day_pred is not None:
    forecast_cat_7day = {
        cat: ratio_pred[i] * seventh_day_pred for i, cat in enumerate(cat_cols)
    }
    pd.DataFrame.from_dict(forecast_cat_7day, orient="index", columns=["予測額"]).to_csv("forecast_category_7day.csv")
    print("📁 7日後のカテゴリ別予測を forecast_category_7day.csv に保存しました")
else:
    print("⚠️ 月末までに7日間がないため、7日後の予測はスキップされました。")
