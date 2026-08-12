# GB wind generation: nowcasting and 6-hour-ahead classification

Classifies Great Britain embedded wind output from weather reanalysis data. Final project for IEE1065 (Machine Learning), Yonsei University summer school, 2026.

## Problem

NESO publishes half-hourly embedded wind generation and capacity for GB; Open-Meteo's archive API provides ERA5-derived hourly weather for four sites across the main onshore wind regions (southern Scotland, northern England, Wales, southwest England). Generation is converted to a capacity factor and merged with site-averaged weather into 8,760 hourly rows for 2025, with no missing values.

Three classification targets of increasing difficulty:

| Target | Question | Classes |
|---|---|---|
| A | Which tercile is output in right now? | low / med / high |
| B | Which tercile will output be in 6 hours? | low / med / high |
| C | Will output fall, hold, or rise over the next 6 hours? | down / flat / up (20/60/20) |

A is a nowcast: it recovers the wind power curve from weather alone, with the current capacity factor excluded from its features. B and C are forecasts and may use the current capacity factor, since it is known at prediction time.

## Results

Random forest (200 trees, depth 8), day-blocked 80/20 split, `random_state=42`:

| Target | Accuracy | Macro F1 | Most-frequent baseline |
|---|---|---|---|
| A: current tercile | 0.851 | 0.84 | 0.316 |
| B: tercile 6h ahead | 0.767 | 0.76 | 0.302 |
| C: ramp direction | 0.634 | 0.47 | 0.611 |

Honest reading: A shows the physics is recoverable (wind speed at 100 m correlates 0.937 with capacity factor). B is the substantive result. C barely beats always predicting "flat"; rebalancing with class weights trades accuracy for ramp recall, examined in section 7.4.

## Leakage controls

Consecutive hours are nearly identical, so a row-level shuffle puts a test hour's neighbour into training. The split therefore blocks by calendar day, and section 5.1 measures what the shuffle would have inflated: up to +0.070 accuracy at full tree depth. Hyperparameter tuning uses `GroupKFold` with the day as the group. Scalers live inside sklearn Pipelines and learn statistics from training folds only.

Known limitations, quantified in `local_audit.py`:

- Tercile boundaries are computed on the full year, so test data participates in the label definition. With 292 random training days the train-only boundaries barely move, but the audit script measures the exact delta.
- For target B, training rows in the last 6 hours of a day take their label from the following day, which can be a test day. Roughly 6% of training rows are affected; the audit script measures the effect of dropping them.
- The most-frequent baseline understates the difficulty of B. The honest comparison is persistence (predict the future tercile equals the current one); the audit script computes it.

## Reproduce

```bash
pip install -r requirements.txt
jupyter notebook wind_forecasting.ipynb
```

Both data sources are public APIs; no credentials or local files needed. Full run takes about 9 minutes, most of it the grid search in section 8. `python local_audit.py` runs the leakage audit standalone.

## Files

```
wind_forecasting.ipynb   the project: collection, cleaning, EDA, models, evaluation, tuning
wind_forecasting.py      script export of the notebook
local_audit.py           standalone leakage audit (reproduction + split experiments)
requirements.txt
```

## Data

- [NESO demand data 2025](https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601) (embedded wind generation and capacity, 17,520 half-hourly rows)
- [Open-Meteo archive API](https://archive-api.open-meteo.com) (ERA5-derived hourly weather, four GB sites, 2025)
