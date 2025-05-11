import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
true_df = pd.read_csv("y_true.csv", parse_dates=["date_time"])
pred_df = pd.read_csv("predicted_total_daily.csv", parse_dates=["date_time"])

# 📆 予測の開始日の前日まで実測値をフィルタリング
forecast_start = pred_df["date_time"].min()
true_df = true_df[true_df["date_time"] < forecast_start]

# 🧮 月ごとに累積
true_df["month"] = true_df["date_time"].dt.to_period("M")
pred_df["month"] = pred_df["date_time"].dt.to_period("M")
true_df["cumsum"] = true_df.groupby("month")["total"].cumsum()
pred_df["cumsum"] = pred_df.groupby("month")["predicted_total"].cumsum()

# 📊 グラフ表示
plt.figure(figsize=(12, 6))
plt.plot(true_df["date_time"], true_df["cumsum"], label="Actual Total (Cumulative)", marker='o')
plt.plot(pred_df["date_time"], pred_df["cumsum"], label="Predicted Total (Cumulative)", linestyle='--', marker='x')
plt.xlabel("日付")
plt.ylabel("累積支出")
plt.title("📆 月別累積支出：実測 vs 予測")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("compare_cumulative_total_by_month.png")
plt.show()
