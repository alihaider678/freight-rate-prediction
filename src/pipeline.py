"""Freight rate model: cleaning, features, time-based validation and final predictions.

Usage:
    python src/pipeline.py validate   # time-based holdout experiments
    python src/pipeline.py final      # train on all data, write submission files
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 42
RPM_BOUNDS = (1.0, 5.0)  # plausible $/mile; rows outside are treated as label errors
HOLDOUT_START = "2025-09-01"
MARKET_WINDOW = 30  # days used for the trailing market baseline
EQUIPMENT = pd.CategoricalDtype(["Dry Van", "Flatbed", "Reefer"])

LGB_PARAMS = dict(
    objective="regression",
    learning_rate=0.03,
    n_estimators=2000,
    num_leaves=63,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, parse_dates=["date"])


def clean(df: pd.DataFrame, weight_medians: pd.Series) -> pd.DataFrame:
    """Fix the data-quality issues found in EDA (applied identically to every split)."""
    df = df.copy()
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["weight"] = df["weight"].abs()  # negative weights are sign errors
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_medians))
    return df


def drop_label_outliers(df: pd.DataFrame) -> pd.DataFrame:
    rpm = df["posted_rate"] / df["distance"]
    return df[rpm.between(*RPM_BOUNDS)]


def market_context(*frames: pd.DataFrame) -> pd.DataFrame:
    """Daily market features built from load features only (no labels), indexed by date.

    EDA showed the per-load market values are noisy and the overall rate level drifts
    independently of market_index, but the daily market_index relative to its recent
    average tracks day-to-day rate moves. quote_signal's effect flips sign with its daily
    level, so both its daily level and the per-load deviation are kept for the trees.
    """
    loads = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    daily = loads.groupby("date").mean().sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max())).interpolate()
    trailing = daily["market_index"].rolling(MARKET_WINDOW, min_periods=1).mean()
    return pd.DataFrame({
        "market_dev": daily["market_index"] - trailing,
        "daily_quote_signal": daily["quote_signal"],
    })


def features(df: pd.DataFrame, context: pd.DataFrame | None, lanes: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["log_distance"] = np.log(df["distance"])
    x["weight"] = df["weight"]
    x["weight_missing"] = df["weight_missing"]
    x["equipment"] = df["equipment"].astype(EQUIPMENT)
    for col in ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]:
        x[col] = df[col]
    x["dlat"] = df["delivery_lat"] - df["pickup_lat"]
    x["dlon"] = df["delivery_lon"] - df["pickup_lon"]
    # Lane level from training rows; unseen cities fall back to lat/lon via NaN
    key = df["pickup"] + "|" + df["delivery"]
    x["lane_rpm"] = key.map(lanes["lane"])
    x["pickup_rpm"] = df["pickup"].map(lanes["pickup"])
    x["delivery_rpm"] = df["delivery"].map(lanes["delivery"])
    x["dayofweek"] = df["date"].dt.dayofweek
    if context is not None:
        x[context.columns] = context.loc[df["date"]].to_numpy()
        x["quote_dev"] = df["quote_signal"] - x["daily_quote_signal"]
    return x


def lane_stats(train: pd.DataFrame, min_loads: int = 5) -> dict:
    """Mean log $/mile residual per lane / city after removing distance and equipment effects."""
    y = target(train)
    r = y - y.groupby([pd.qcut(train["distance"], 20), train["equipment"]], observed=True).transform("mean")
    key = train["pickup"] + "|" + train["delivery"]
    out = {}
    for name, group in [("lane", key), ("pickup", train["pickup"]), ("delivery", train["delivery"])]:
        stats = r.groupby(group).agg(["mean", "size"])
        out[name] = stats.loc[stats["size"] >= min_loads, "mean"]
    return out


def target(df: pd.DataFrame) -> pd.Series:
    return np.log(df["posted_rate"] / df["distance"])  # log $/mile removes the dominant distance scale


class RateModel:
    def __init__(self, use_market: bool = True, n_estimators: int | None = None):
        self.use_market = use_market
        self.params = {**LGB_PARAMS, **({"n_estimators": n_estimators} if n_estimators else {})}

    def fit(self, train: pd.DataFrame, context: pd.DataFrame, eval_df: pd.DataFrame | None = None):
        self.context = context if self.use_market else None
        self.lanes = lane_stats(train)
        self.model = lgb.LGBMRegressor(**self.params)
        kwargs = {}
        if eval_df is not None:
            kwargs = dict(eval_X=self._x(eval_df), eval_y=target(eval_df),
                          callbacks=[lgb.early_stopping(100, verbose=False)])
        self.model.fit(self._x(train), target(train), **kwargs)
        return self

    def _x(self, df: pd.DataFrame) -> pd.DataFrame:
        return features(df, self.context, self.lanes)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.model.predict(self._x(df))) * df["distance"].to_numpy()

    def predict_marginal(self, df: pd.DataFrame, reference: pd.DataFrame) -> np.ndarray:
        """Predict rows that have no per-load quote_signal (the December chart).

        The quote effect is non-linear, so instead of plugging in one value we average the
        prediction over every quote_signal observed on that date in `reference`.
        """
        rows = df.drop(columns="quote_signal", errors="ignore").reset_index(names="_row")
        expanded = rows.merge(reference[["date", "quote_signal"]], on="date")
        expanded["_pred"] = self.predict(expanded)
        return expanded.groupby("_row")["_pred"].mean().reindex(rows["_row"]).to_numpy()


def metrics(y: pd.Series, p: np.ndarray) -> dict:
    err = p - y.to_numpy()
    return {
        "MAE": round(float(np.mean(np.abs(err))), 2),
        "RMSE": round(float(np.sqrt(np.mean(err**2))), 2),
        "MAPE%": round(float(np.mean(np.abs(err) / y)) * 100, 2),
    }


def validate() -> None:
    raw = load("train_test.csv")
    is_holdout = raw["date"] >= HOLDOUT_START
    weight_medians = raw.loc[~is_holdout, "weight"].abs().groupby(raw["equipment"]).median()
    data = clean(raw, weight_medians)
    train = drop_label_outliers(data[~is_holdout])
    hold = data[is_holdout]
    clean_hold = drop_label_outliers(hold)
    context = market_context(raw)  # features only; trailing window never looks ahead
    print(f"train {len(train):,} rows (Jan-Aug, outliers removed) | holdout {len(hold):,} rows (Sep-Oct)")

    # Pick the number of trees with early stopping on the last training month
    inner_tr, inner_va = train[train["date"] < "2025-08-01"], train[train["date"] >= "2025-08-01"]
    best = {m: RateModel(m).fit(inner_tr, context, inner_va).model.best_iteration_ for m in (False, True)}
    print("best n_estimators (no market, market):", best)

    rpm = (train["posted_rate"] / train["distance"]).groupby(train["equipment"]).median()
    no_market = RateModel(False, best[False]).fit(train, context)
    market = RateModel(True, best[True]).fit(train, context)
    candidates = {
        "baseline: median $/mile x distance": hold["equipment"].map(rpm).to_numpy() * hold["distance"].to_numpy(),
        "LightGBM, no market signals": no_market.predict(hold),
        "LightGBM, market context": market.predict(hold),
        "LightGBM, market context, quote marginalised": market.predict_marginal(hold, hold),
    }
    keep = hold.index.isin(clean_hold.index)
    rows = []
    for name, p in candidates.items():
        rows.append({"model": name, **metrics(hold["posted_rate"], p),
                     **{f"{k} (clean)": v for k, v in metrics(clean_hold["posted_rate"], p[keep]).items()}})
    print(pd.DataFrame(rows).to_string(index=False))

    imp = pd.Series(market.model.booster_.feature_importance("gain"), index=market.model.feature_name_)
    print("\nfeature importance (gain %):", (imp / imp.sum() * 100).sort_values(ascending=False).round(1).to_dict())


def december_rows(template: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Turn the 7-column December chart template into full model rows."""
    rows = template.drop(columns="predicted_rate").copy()
    rows["date"] = pd.to_datetime(rows["date"])
    coords = {
        "pickup": train.groupby("pickup")[["pickup_lat", "pickup_lon"]].first(),
        "delivery": train.groupby("delivery")[["delivery_lat", "delivery_lon"]].first(),
    }
    for side, table in coords.items():
        rows = rows.join(table, on=side)
    rows["weight_missing"] = 0
    return rows


def final() -> None:
    raw = load("train_test.csv")
    valid_raw = load("validation.csv")
    weight_medians = raw["weight"].abs().groupby(raw["equipment"]).median()
    data = drop_label_outliers(clean(raw, weight_medians))
    valid = clean(valid_raw, weight_medians)
    context = market_context(raw, valid_raw)  # continuous Jan-Dec daily market features

    # Same tree-count procedure as validation, now with Sep-Oct as the early-stopping window
    is_recent = data["date"] >= HOLDOUT_START
    n_trees = RateModel(True).fit(data[~is_recent], context, data[is_recent]).model.best_iteration_
    model = RateModel(True, n_trees).fit(data, context)
    print(f"trained on {len(data):,} rows with {n_trees} trees")

    submission = pd.read_csv(DATA / "validation_predictions_template.csv")
    predicted = pd.Series(model.predict(valid), index=valid["load_id"])
    submission["predicted_rate"] = submission["load_id"].map(predicted).round(2)
    submission.to_csv(ROOT / "validation_predictions.csv", index=False)
    print("wrote validation_predictions.csv:", submission["predicted_rate"].describe().round(1).to_dict())

    template_path = DATA / "december_chart_inputs.csv"
    template = pd.read_csv(template_path)
    rows = december_rows(template, raw)
    template["predicted_rate"] = model.predict_marginal(rows, valid).round(2)
    template.to_csv(template_path, index=False)
    print("wrote december_chart_inputs.csv:", template["predicted_rate"].describe().round(1).to_dict())


if __name__ == "__main__":
    {"validate": validate, "final": final}[sys.argv[1] if len(sys.argv) > 1 else "validate"]()
