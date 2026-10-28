# Freight Rate Prediction

Predicts the posted rate of a truckload from lane, distance, equipment, weight, date and market signals. Built for the Spotter Machine Learning Engineer assessment.

The model is a LightGBM regressor trained on the log of rate per mile. It is validated on a time-based holdout (train on January to August 2025, evaluate on September to October 2025), which mirrors the real task of predicting November and December loads from earlier history.

## Results

Holdout: September to October 2025, 9,523 loads.

| Model | MAE ($) | RMSE ($) | MAPE |
|---|---|---|---|
| Baseline: median $/mile by equipment x distance | 229.12 | 670.67 | 10.49% |
| LightGBM without market signals | 100.88 | 630.93 | 4.43% |
| LightGBM with market context (final) | 96.03 | 628.54 | 4.27% |

Excluding the roughly 1% of holdout rows with implausible labels (below $1 or above $5 per mile), the final model's MAE is $43.29 and its MAPE is 1.79%. The high RMSE on all rows comes almost entirely from those label errors.

## Repository layout

```
data/                       Input data provided with the assessment
  train_test.csv            48,000 labelled loads, Jan to Oct 2025
  validation.csv            12,000 loads to predict, Nov to Dec 2025
  december_chart_inputs.csv Fixed-lane December rows (filled by the pipeline)
src/
  eda.py                    Data quality and pattern checks
  pipeline.py               Cleaning, features, validation and final training
score.py                    Scorer provided by Spotter (unchanged)
validation_predictions.csv  Final submission: load_id, predicted_rate
scorer_results/             December chart produced by score.py
reports/                    Assessment report
```

## Setup

Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Running

```bash
python src/eda.py                 # data quality summary
python src/pipeline.py validate   # time-based holdout comparison
python src/pipeline.py final      # train on all data, write both prediction files
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`final` writes `validation_predictions.csv` and fills the `predicted_rate` column of `data/december_chart_inputs.csv`. The scorer then validates both files and saves `scorer_results/candidate_december.png`. A full run takes under a minute on a laptop.

## Approach

**Data cleaning**

- 292 negative weights in training (145 in validation) are treated as sign errors and made positive.
- Missing weights (300 train, 165 validation) are filled with the median for the equipment type, and a missing flag is kept as a feature.
- Missing `market_index` values (374 train, 249 validation) are not used directly. The model relies on daily market averages instead (see below).
- About 650 training loads have a rate per mile below $1 or above $5, up to $14 per mile. These look like data-entry errors and are removed from training only. They are kept in the holdout so the reported metrics are honest.

**Target**

Distance explains most of the rate (correlation 0.91), which hides every other effect. The model predicts `log(posted_rate / distance)` and multiplies back by distance. The log scale also makes the model optimise relative error, which suits rates ranging from under $100 to over $6,000.

**Features**

- Distance (log), equipment, weight, pickup and delivery coordinates and their differences.
- Lane, pickup and delivery premiums: the average log rate per mile of each lane and city after removing distance and equipment effects. These are computed from training rows only. 12% of validation loads use lanes, and 8 cities, that never appear in training; for those the premium is missing and the model falls back on coordinates.
- Market context, described next.

**Market signals**

Using `market_index` and `quote_signal` as raw per-load values made the holdout worse, so I looked at how they behave over time:

- The daily average `market_index` tracks the daily rate level closely (correlation 0.88 from January to August), but the overall rate level drifts in a way the index does not explain. A model that maps the index level directly to price is biased on later months. What carries over is the index relative to its own trailing 30-day average, which explains day-to-day movement. This is the `market_dev` feature.
- The effect of `quote_signal` changes sign with its daily level. On days where the average is around 2.2 it correlates positively with rate (+0.3), and around 1.93 it correlates negatively (-0.3). The model receives both the day's average quote signal and each load's deviation from it, so the trees can learn the regime. In November and December the daily average stays near 2.05, the neutral regime.

All market features are built from load features only, never from labels, so using validation-period values is legitimate.

**December chart**

The December input file has no `market_index` or `quote_signal`. For each December date the pipeline takes that day's market context from `validation.csv` and averages the model's prediction over every `quote_signal` observed on that date. Averaging predictions, rather than plugging in one average signal, matters because the quote effect is non-linear. On the September to October holdout, a daily curve built this way for the same fixed lane had a correlation of 0.95 with the actual market-wide daily rate per mile, against 0.41 for a model without market signals.

**Final model**

The number of trees is chosen by early stopping on September to October. The model is then refit on all cleaned January to October data and used for both prediction files.

## Limitations

- The overall rate level for November and December cannot be confirmed from the data. The model assumes it stays near the January to October average, adjusted by the market context.
- Unseen cities rely on coordinates alone, so their predictions are less precise than those for known lanes.
