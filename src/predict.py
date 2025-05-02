# src/predict.py
import joblib, pandas as pd, numpy as np, datetime as dt
from pathlib import Path
from utils import load_data

# ---------- パス設定 ----------
BASE_DIR  = Path(__file__).resolve().parents[1]
DATA_CSV  = BASE_DIR / "data" / "spending_cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR   = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)

# ---------- 1) 総額（Prophet）14 日予測 ----------
prophet = joblib.load(MODEL_DIR / "prophet_total.pkl")

# 予測日程は “最後の学習日翌日” から 14 日
future = prophet.make_future_dataframe(periods=14, freq="D", include_history=False)
total_fcst = prophet.predict(future)[["ds", "yhat"]]
total_fcst["ds"] = pd.to_datetime(total_fcst["ds"]).dt.tz_localize(None).dt.normalize()

# ---------- 2) カテゴリ（LightGBM）14 日予測 ----------
gbm = joblib.load(MODEL_DIR / "lgbm_category.pkl")
df  = load_data(DATA_CSV)                        # cleaned raw
df["date"] = df["date_time"].dt.date

# Prophet が出力した “未来 14 日” をそのまま使う
forecast_days = total_fcst["ds"]                 # Timestamp 14 個
cat_list      = df["category"].unique()

records = []
for ts in forecast_days:
    d_date = ts.date()                           # datetime.date
    for cat in cat_list:
        # 過去シリーズ（カテゴリ別の日次合計）
        series = (df[df["category"] == cat]
                    .groupby("date")["amount"]
                    .sum())
        med  = series.median()
        lag1 = series.get(d_date - dt.timedelta(days=1), med)
        lag7 = series.get(d_date - dt.timedelta(days=7), med)

        feat = pd.DataFrame({
            "dow":   [ts.weekday()],
            "month": [ts.month],
            "lag1":  [lag1],
            "lag7":  [lag7],
        })
        amt = gbm.predict(feat)[0]
        records.append({"ds": ts, "category": cat, "amount_pred": amt})

cat_fcst = pd.DataFrame(records)
cat_fcst["ds"] = pd.to_datetime(cat_fcst["ds"]).dt.tz_localize(None).dt.normalize()

# ---------- 3) トップダウン整合 ----------
def reconcile(df_cat: pd.DataFrame, df_tot: pd.DataFrame) -> pd.DataFrame:
    tot_map = dict(zip(df_tot["ds"], df_tot["yhat"]))  # ds → 総額

    out = []
    for d, g in df_cat.groupby("ds"):
        if d not in tot_map:
            raise ValueError(f"Total forecast missing for {d}")
        total_hat = tot_map[d]
        share = g["amount_pred"] / g["amount_pred"].sum()
        out.extend(g.assign(amount_final = share * total_hat).to_dict("records"))
    return pd.DataFrame(out)

result_df = reconcile(cat_fcst, total_fcst)

# ---------- 4) 出力 ----------
out_path = OUT_DIR / "forecast_14d.csv"
result_df.to_csv(out_path, index=False, encoding="utf-8")
print(f"✅ Forecast written → {out_path}")
