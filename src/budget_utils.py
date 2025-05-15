import pandas as pd
import numpy as np
from datetime import datetime

def rebalance_budget(predicted, budget_dict):
    df = pd.DataFrame({"カテゴリ": list(predicted.keys()), "予測額": list(predicted.values())})
    df["予算"] = df["カテゴリ"].map(budget_dict)
    df["残予算"] = df["予算"] - df["予測額"]
    df["不足額"] = df["予測額"] - df["予算"]
    df["不足額"] = df["不足額"].apply(lambda x: x if x > 0 else 0)
    df["余剰可能額"] = df["残予算"].apply(lambda x: x if x > 0 else 0)

    shortfall_df = df[df["不足額"] > 0].copy()
    surplus_df = df[df["余剰可能額"] > 0].copy()

    total_shortfall = shortfall_df["不足額"].sum()
    total_surplus = surplus_df["余剰可能額"].sum()

    if total_surplus > 0:
        shortfall_df["補填額"] = shortfall_df["不足額"].apply(
            lambda x: min(x, total_surplus * (x / total_shortfall))
        )
    else:
        shortfall_df["補填額"] = 0

    return shortfall_df[["カテゴリ", "予算", "予測額", "補填額"]]

def calc_daily_limits(budget_dict):
    today = datetime.today()
    end_of_month = today.replace(day=1) + pd.offsets.MonthEnd(1)
    remaining_days = (end_of_month - today).days

    category_limits = {
        cat: {
            "残日数支出上限": budget_dict[cat],
            "1日あたり上限": budget_dict[cat] / remaining_days if remaining_days > 0 else 0
        } for cat in budget_dict
    }

    return pd.DataFrame.from_dict(category_limits, orient="index").reset_index().rename(columns={"index": "カテゴリ"})

def simulate_with_user_budget(model, X_test, cat_cols, user_budget_dict):
    model.eval()
    with torch.no_grad():
        pred_ratios = model(X_test).numpy()

    avg_ratios = pred_ratios.mean(axis=0)
    risk_report = []

    for i, cat in enumerate(cat_cols):
        predicted_ratio = avg_ratios[i]
        predicted_amount = predicted_ratio * user_budget_dict.get("total_budget", 100000)
        budget = user_budget_dict.get(cat, predicted_amount)

        if predicted_amount > budget * 1.2:
            risk = "🔴 高リスク"
        elif predicted_amount > budget:
            risk = "🟠 中リスク"
        else:
            risk = "🟢 低リスク"

        risk_report.append((cat, predicted_amount, budget, risk))

    print("\n🧮 シミュレーション結果：予算に基づくリスク評価")
    print("カテゴリ\t予測額\t予算\tリスク")
    for cat, pred_amt, budget, risk in risk_report:
        print(f"{cat}\t{pred_amt:,.0f}\t{budget:,.0f}\t{risk}")

def calculate_last_month_budget(df, target_month):
    df["date_time"] = pd.to_datetime(df["date_time"], errors="coerce")
    if df["date_time"].dt.tz is not None:
        df["date_time"] = df["date_time"].dt.tz_localize(None)
    df["month"] = df["date_time"].dt.to_period("M").astype(str)
    last_month_df = df[df["month"] == target_month]

    if last_month_df.empty:
        print(f"⚠️ {target_month} のデータが見つかりません。すべて0として扱います。")
        return {
            "total_budget": 0,
            "food": 0,
            "transport": 0,
            "entertainment": 0,
            "clothing_beauty_daily": 0,
            "utilities": 0,
            "social": 0,
            "other": 0,
        }

    result = last_month_df.groupby("category")["amount"].sum().to_dict()
    total = sum(result.values())
    result = {k: float(v) for k, v in result.items()}
    result["total_budget"] = total
    return result
