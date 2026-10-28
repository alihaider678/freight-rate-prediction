"""Quick data-quality and pattern checks for the freight rate data."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

train = pd.read_csv(DATA / "train_test.csv", parse_dates=["date"])
valid = pd.read_csv(DATA / "validation.csv", parse_dates=["date"])

for name, df in [("train", train), ("valid", valid)]:
    print(f"\n===== {name} {df.shape} {df['date'].min().date()} -> {df['date'].max().date()}")
    print("nulls:", df.isna().sum()[lambda s: s > 0].to_dict())
    print("dup load_id:", df["load_id"].duplicated().sum(),
          "| dup rows (excl id):", df.drop(columns="load_id").duplicated().sum())
    print(df.describe().T[["min", "50%", "max"]].round(3).to_string())
    for col in ["equipment", "pickup", "delivery"]:
        vals = df[col].value_counts(dropna=False)
        print(f"{col}: {len(vals)} unique ->", vals.head(8).to_dict())

# Labels and per-mile rate
t = train.copy()
t["rpm"] = t["posted_rate"] / t["distance"]
print("\nrate-per-mile quantiles:", t["rpm"].quantile([0, .01, .5, .99, 1]).round(3).to_dict())
print("non-positive posted_rate:", (t["posted_rate"] <= 0).sum())

# Categories unseen in training
for col in ["pickup", "delivery", "equipment"]:
    print(f"valid {col} unseen in train:", sorted(set(valid[col].dropna()) - set(train[col].dropna()))[:10])
lanes_tr = set(zip(train["pickup"], train["delivery"]))
print("valid lanes unseen in train:", (~pd.Series(list(zip(valid["pickup"], valid["delivery"]))).isin(lanes_tr)).mean().round(3))
print("Lexington->Fort Wayne in train:", ((train["pickup"] == "Lexington") & (train["delivery"] == "Fort Wayne")).sum())

# Same city name with different coordinates (geo consistency)
print("pickup cities w/ >1 coordinate:", (train.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique().max(axis=1) > 1).sum())

# Correlations and time patterns
print("\ncorr with posted_rate:\n", t.select_dtypes("number").corr()["posted_rate"].round(3).to_string())
print("corr with rpm:\n", t.select_dtypes("number").corr()["rpm"].round(3).to_string())
print("\nmonthly mean rpm:", t.groupby(t["date"].dt.month)["rpm"].mean().round(3).to_dict())
print("weekday mean rpm:", t.groupby(t["date"].dt.dayofweek)["rpm"].mean().round(3).to_dict())
print("equipment mean rpm:", t.groupby("equipment")["rpm"].mean().round(3).to_dict())

# Market signals: are they daily (shared by date) or per load?
for col in ["market_index", "quote_signal"]:
    print(f"{col} unique per date (mean):", train.groupby("date")[col].nunique().mean().round(2))
print("valid daily market_index (Dec head):",
      valid[valid["date"].dt.month == 12].groupby("date")["market_index"].mean().head(5).round(3).to_dict())
