"""Local leakage audit for the wind notebook. Run on WSL2: python3 local_audit.py

Q1: tercile thresholds computed on the full year (test data included in label
    definition). Recompute on train only; measure the delta.
Q2: random-day split vs chronological split (regime correlation between
    adjacent days). Measure the delta.
Q3: target B labels reaching across the day boundary into test days
    (cf_future = shift(-6)). Drop affected train rows; measure the delta.
Plus: persistence baseline for B, shuffled-target sanity check for A.
"""

import numpy as np
import pandas as pd
import requests
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

RS = 42
H = 6

# ---------------------------------------------------------------- data
URL_2025 = ("https://api.neso.energy/dataset/8f2fe0af-871c-488d-8bad-960426f24601/"
            "resource/b2bde559-3455-4021-b179-dfe60c0337b0/download/demanddata_2025.csv")
neso = pd.read_csv(URL_2025)
print("NESO rows:", len(neso))

SITES = {
    "s_scotland": (55.5, -4.0),
    "n_england":  (54.5, -2.0),
    "wales":      (52.5, -3.5),
    "sw_england": (50.8, -4.0),
}
frames = []
for name, (lat, lon) in SITES.items():
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon,
        "start_date": "2025-01-01", "end_date": "2025-12-31",
        "hourly": ("wind_speed_100m,wind_speed_10m,temperature_2m,"
                   "surface_pressure,cloud_cover,shortwave_radiation,"
                   "wind_direction_100m,wind_gusts_10m"),
        "wind_speed_unit": "ms", "timezone": "UTC",
    }, timeout=120)
    r.raise_for_status()
    df_site = pd.DataFrame(r.json()["hourly"])
    df_site["site"] = name
    frames.append(df_site)
    print(name, df_site.shape)

weather = pd.concat(frames, ignore_index=True)
weather["time"] = pd.to_datetime(weather["time"], utc=True)

day_start = (pd.to_datetime(neso["SETTLEMENT_DATE"])
             .dt.tz_localize("Europe/London").dt.tz_convert("UTC"))
neso["time"] = day_start + pd.to_timedelta((neso["SETTLEMENT_PERIOD"] - 1) * 30, unit="m")

n = neso[["time", "EMBEDDED_WIND_GENERATION", "EMBEDDED_WIND_CAPACITY"]].copy()
n["wind_cf"] = n["EMBEDDED_WIND_GENERATION"] / n["EMBEDDED_WIND_CAPACITY"]
n_hourly = n.set_index("time").resample("h").mean().reset_index()
w_hourly = weather.drop(columns=["site"]).groupby("time").mean().reset_index()
df = pd.merge(n_hourly, w_hourly, on="time", how="inner")
print("merged:", df.shape)

# ---------------------------------------------------------------- features
d = df.sort_values("time").reset_index(drop=True).copy()
rad = np.deg2rad(d["wind_direction_100m"])
d["wind_dir_sin"] = np.sin(rad)
d["wind_dir_cos"] = np.cos(rad)
d["wind_speed_100m_cubed"] = d["wind_speed_100m"] ** 3
d["gust_ratio"] = d["wind_gusts_10m"] / d["wind_speed_10m"]
d["shear_ratio"] = d["wind_speed_100m"] / d["wind_speed_10m"]
hour = d["time"].dt.hour
doy = d["time"].dt.dayofyear
d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
d["doy_sin"] = np.sin(2 * np.pi * doy / 365)
d["doy_cos"] = np.cos(2 * np.pi * doy / 365)

d["cf_future"] = d["wind_cf"].shift(-H)
d["cf_change"] = d["cf_future"] - d["wind_cf"]
d["y_now"] = pd.qcut(d["wind_cf"], 3, labels=["low", "med", "high"])
d["y_future"] = pd.qcut(d["cf_future"], 3, labels=["low", "med", "high"])
lo, hi = d["cf_change"].quantile([0.20, 0.80])
d["y_ramp"] = pd.cut(d["cf_change"], bins=[-np.inf, lo, hi, np.inf],
                     labels=["down", "flat", "up"])
d = d.dropna(subset=["cf_future"]).reset_index(drop=True)

FEATURES = [
    "wind_speed_100m", "wind_speed_10m", "wind_gusts_10m", "wind_speed_100m_cubed",
    "surface_pressure", "temperature_2m", "cloud_cover", "shortwave_radiation",
    "wind_dir_sin", "wind_dir_cos", "gust_ratio", "shear_ratio",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos", "wind_cf",
]
FEATURES_NOW = [f for f in FEATURES if f != "wind_cf"]

d["day"] = d["time"].dt.date
days = d["day"].unique()
train_days, test_days = train_test_split(days, test_size=0.2, random_state=RS)
train_days, test_days = set(train_days), set(test_days)
train_mask = d["day"].isin(train_days)
test_mask = d["day"].isin(test_days)


def rf():
    return RandomForestClassifier(n_estimators=200, max_depth=8,
                                  random_state=RS, n_jobs=-1)


def run(target, feats, tr_mask, te_mask, y=None):
    y = d[target] if y is None else y
    m = rf().fit(d.loc[tr_mask, feats], y[tr_mask])
    pred = m.predict(d.loc[te_mask, feats])
    acc = (pred == y[te_mask].astype(str)).mean()
    return acc, f1_score(y[te_mask].astype(str), pred, average="macro")


print("\n================ REPRODUCTION (notebook pipeline, verbatim) ================")
for target, feats, label in [("y_now", FEATURES_NOW, "A current tercile"),
                             ("y_future", FEATURES, "B tercile 6h ahead"),
                             ("y_ramp", FEATURES, "C ramp direction")]:
    dummy = DummyClassifier(strategy="most_frequent").fit(
        d.loc[train_mask, feats], d.loc[train_mask, target])
    b = dummy.score(d.loc[test_mask, feats], d.loc[test_mask, target])
    acc, f1 = run(target, feats, train_mask, test_mask)
    print(f"{label:<22} acc={acc:.3f}  macroF1={f1:.3f}  baseline={b:.3f}")

print("\n================ Q1: tercile thresholds from train only ================")
cf_train = d.loc[train_mask, "wind_cf"]
edges_now = cf_train.quantile([1 / 3, 2 / 3]).to_numpy()
y_now_tr = pd.cut(d["wind_cf"], bins=[-np.inf, *edges_now, np.inf],
                  labels=["low", "med", "high"])
acc, f1 = run("y_now", FEATURES_NOW, train_mask, test_mask, y=y_now_tr)
print(f"A with train-only thresholds: acc={acc:.3f}  macroF1={f1:.3f}")
print(f"full-year edges vs train-only edges: "
      f"{d['wind_cf'].quantile([1/3, 2/3]).round(4).tolist()} vs {np.round(edges_now, 4).tolist()}")

print("\n================ Q2: chronological day split ================")
days_sorted = sorted(days)
cut = int(len(days_sorted) * 0.8)
chron_train = d["day"].isin(set(days_sorted[:cut]))
chron_test = d["day"].isin(set(days_sorted[cut:]))
for target, feats, label in [("y_now", FEATURES_NOW, "A"), ("y_future", FEATURES, "B")]:
    acc, f1 = run(target, feats, chron_train, chron_test)
    print(f"{label} chronological (train Jan-Oct, test Oct-Dec): acc={acc:.3f}  macroF1={f1:.3f}")
print("note: chronological split also shifts the season mix; delta is an upper bound")

print("\n================ Q3: target B labels crossing into test days ================")
future_day = d["time"].add(pd.Timedelta(hours=H)).dt.date
contaminated = train_mask & future_day.isin(test_days)
print(f"train rows whose 6h-ahead label lands on a test day: {int(contaminated.sum())} "
      f"of {int(train_mask.sum())}")
clean_train = train_mask & ~contaminated
acc, f1 = run("y_future", FEATURES, clean_train, test_mask)
print(f"B with contaminated train rows dropped: acc={acc:.3f}  macroF1={f1:.3f}")

print("\n================ Persistence baseline for B ================")
pers_pred = d.loc[test_mask, "y_now"].astype(str)
pers_actual = d.loc[test_mask, "y_future"].astype(str)
print(f"predict future tercile = current tercile: acc={(pers_pred == pers_actual).mean():.3f}  "
      f"macroF1={f1_score(pers_actual, pers_pred, average='macro'):.3f}")

print("\n================ Shuffled-target sanity check for A ================")
rng = np.random.default_rng(0)
y_shuf = pd.Series(rng.permutation(d["y_now"].astype(str).to_numpy()), index=d.index)
m = rf().fit(d.loc[train_mask, FEATURES_NOW], y_shuf[train_mask])
acc = (m.predict(d.loc[test_mask, FEATURES_NOW]) == y_shuf[test_mask]).mean()
print(f"A on permuted labels: acc={acc:.3f} (chance ~0.333)")
