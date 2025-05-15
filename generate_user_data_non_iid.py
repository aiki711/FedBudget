import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# --- 設定 ---
n_users = 3
days = 150  # 約5ヶ月分のデータ
base_date = datetime(2025, 1, 1)
PAYDAY = 25

# --- 全カテゴリ ---
all_categories = ["food", "transport", "entertainment", "clothing_beauty_daily", "utilities", "social", "other"]

# --- ユーザー別カテゴリ重み + 残りを均等に薄く加える ---
base_preferences = {
    "U001": {"food": 0.5, "transport": 0.3, "utilities": 0.2},
    "U002": {"entertainment": 0.4, "social": 0.3, "other": 0.3},
    "U003": {"clothing_beauty_daily": 0.4, "food": 0.3, "other": 0.3},
}

user_preferences = {}
for uid, prefs in base_preferences.items():
    weights = prefs.copy()
    missing_cats = [cat for cat in all_categories if cat not in weights]
    for mc in missing_cats:
        weights[mc] = 0.01  # ごく小さく追加
    total = sum(weights.values())
    normalized = {k: v / total for k, v in weights.items()}
    user_preferences[uid] = normalized
    
# --- ディレクトリ準備 ---
base_dir = Path("data")
user_dir = base_dir / "users"
base_dir.mkdir(exist_ok=True)
user_dir.mkdir(exist_ok=True)

# --- ユーザーIDリスト ---
user_ids = list(user_preferences.keys())

# --- データ生成 ---
records = []

for uid in user_ids:
    cat_weights = user_preferences[uid]
    cats = list(cat_weights.keys())
    probs = [cat_weights[c] for c in cats]

    for i in range(days):
        date = base_date + timedelta(days=i)
        weekday = date.weekday()
        is_weekend = weekday >= 5
        is_holiday = random.random() < 0.05  # 5%確率で祝日

        # 支出件数（1日あたり1〜3カテゴリ）
        n_spends = random.randint(1, 3)
        chosen_categories = np.random.choice(cats, size=n_spends, p=probs, replace=False)

        for cat in chosen_categories:
            amount = round(np.random.exponential(scale=5000), -1)  # 平均5000円程度
            day = date.day
            days_from_payday = (date - date.replace(day=PAYDAY)).days
            if day < PAYDAY:
                days_from_payday = day - PAYDAY
            else:
                days_from_payday = day - PAYDAY

            records.append({
                "date_time": date.isoformat() + "+09:00",
                "user_id": uid,
                "category": cat,
                "days_from_payday": days_from_payday,
                "weekday": weekday,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "amount": amount
            })

# --- DataFrame化 ---
df = pd.DataFrame(records)
df = df.sort_values(["user_id", "date_time"]).reset_index(drop=True)

# --- 保存 ---
df.to_csv(base_dir / "spending_cleaned.csv", index=False)
print(f"✅ spending_cleaned.csv saved to: {base_dir / 'spending_cleaned.csv'}")

# --- ユーザーごとに分割保存 ---
for uid in user_ids:
    df_user = df[df["user_id"] == uid]
    df_user.to_csv(user_dir / f"{uid}.csv", index=False)
    print(f"👤 {uid}.csv saved to: {user_dir / f'{uid}.csv'}")
