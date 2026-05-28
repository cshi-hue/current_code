import pandas as pd
import numpy as np
import lightgbm as lgb

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
from dateutil.easter import easter
from datetime import datetime, timedelta

################### shared setup ###########################
offset = 0
today = datetime.today() - timedelta(days=offset)
today_str = today.strftime("%m%d")
run_str = today.strftime("%y%m%d")
run_dt = pd.to_datetime(run_str, format="%y%m%d")

weather_vars = [
    "temperature_2m",
    "apparent_temperature",
    "dew_point_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "pressure_msl",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "global_tilted_irradiance",
    "direct_normal_irradiance",
    "terrestrial_radiation"
]

################### load forecast ###########################

load_forecast = pd.read_csv(
    f"data/load_frcstd_7_day/load_frcstd_7_day_{today_str}.csv",
    skiprows=[1]
)

load_forecast["forecast_load_mw"] = pd.to_numeric(
    load_forecast["forecast_load_mw"], errors="coerce"
)

load_forecast = load_forecast[
    ["evaluated_at_datetime_ept", "forecast_datetime_ending_ept", "forecast_area", "forecast_load_mw"]
].copy()

load_forecast["forecast_datetime_ending_ept"] = pd.to_datetime(
    load_forecast["forecast_datetime_ending_ept"]
)

forecast_to_agg = {
    "AE/MIDATL": "AE",
    "AEP": "AEP",
    "AP": "AP",
    "ATSI": "ATSI",
    "BG&E/MIDATL": "BGE",
    "COMED": "COMED",
    "DAYTON": "DAYTON",
    "DEOK": "DEOK",
    "DOMINION": "DOM",
    "DP&L/MIDATL": "DPL",
    "DUQUESNE": "DUQ",
    "EKPC": "EKPC",
    "JCP&L/MIDATL": "JC",
    "METED/MIDATL": "METED",
    "PECO/MIDATL": "PE",
    "PENELEC/MIDATL": "PN",
    "PEPCO/MIDATL": "PEP",
    "PPL/MIDATL": "PL",
    "PSE&G/MIDATL": "PS",
    "RECO/MIDATL": "RE",
    "UGI/MIDATL": "UGI",
    "WESTERN_REGION": "WEST_REGION",
    "SOUTHERN_REGION": "SOUTH_REGION",
    "MID_ATLANTIC_REGION": "MIDATL_REGION",
    "RTO_COMBINED": "RTO"
}

load_forecast["agg_NodeName"] = load_forecast["forecast_area"].map(forecast_to_agg)

load_forecast = (
    load_forecast
    .sort_values(["forecast_datetime_ending_ept", "agg_NodeName", "evaluated_at_datetime_ept"])
    .drop_duplicates(subset=["forecast_datetime_ending_ept", "agg_NodeName"], keep="last")
    .reset_index(drop=True)
)

rto_load_forecast = load_forecast[load_forecast["agg_NodeName"] == "RTO"].copy()

################### weather data ########################

hist_files = [
    "data/weather/weather_hist_2023.csv",
    "data/weather/weather_hist_2024.csv",
    "data/weather/weather_hist_2025.csv",
    "data/weather/weather_hist_2026.csv",
]

forecast_file = f"data/weather/weather_forecast_{run_str}.csv"

# --- load historical ---
hist_weather = pd.concat([pd.read_csv(f) for f in hist_files], ignore_index=True)
hist_weather.drop(columns=["weather_code"], errors="ignore", inplace=True)
hist_weather.rename(columns={"time": "datetime_ending_ept"}, inplace=True)
hist_weather["datetime_ending_ept"] = pd.to_datetime(hist_weather["datetime_ending_ept"])
hist_weather = hist_weather[hist_weather["datetime_ending_ept"] <= run_dt].copy()

# --- load forecast ---
forecast_weather = pd.read_csv(forecast_file)
forecast_weather.drop(columns=["weather_code"], errors="ignore", inplace=True)
forecast_weather.rename(columns={"time": "datetime_ending_ept"}, inplace=True)
forecast_weather["datetime_ending_ept"] = pd.to_datetime(forecast_weather["datetime_ending_ept"])
forecast_weather = forecast_weather[forecast_weather["datetime_ending_ept"] > run_dt].copy()

# --- dedupe ---
dedupe_cols = ["zone", "weather_station", "lat", "lon", "datetime_ending_ept"]
hist_weather = hist_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
forecast_weather = forecast_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

# --- fill forecast nulls with hour average ---
forecast_weather["hour"] = forecast_weather["datetime_ending_ept"].dt.hour

hourly_avg = (
    forecast_weather
    .groupby(["zone", "weather_station", "hour"])[weather_vars]
    .mean()
    .reset_index()
)

forecast_weather = forecast_weather.merge(
    hourly_avg,
    on=["zone", "weather_station", "hour"],
    how="left",
    suffixes=("", "_hour_avg")
)

for col in weather_vars:
    forecast_weather[col] = forecast_weather[col].fillna(forecast_weather[f"{col}_hour_avg"])

forecast_weather.drop(
    columns=["hour"] + [f"{c}_hour_avg" for c in weather_vars],
    inplace=True,
    errors="ignore"
)

# --- final zonal_weather ---
zonal_weather = pd.concat([hist_weather, forecast_weather], ignore_index=True)
zonal_weather = zonal_weather.sort_values(
    ["zone", "weather_station", "datetime_ending_ept"]
).reset_index(drop=True)

for var in weather_vars:
    zonal_weather[f"{var}_weighted"] = zonal_weather[var] * zonal_weather["weather_weight"]

zonal_weather = (
    zonal_weather.groupby(["datetime_ending_ept", "zone"])[[f"{v}_weighted" for v in weather_vars]]
    .sum()
    .reset_index()
)

zonal_weather = zonal_weather.rename(
    columns={f"{col}_weighted": col for col in weather_vars}
)

zone_map = {
    "AE": "AE",
    "AEP": "AEP",
    "APS": "AP",
    "ATSI": "ATSI",
    "BGE": "BGE",
    "COMED": "COMED",
    "DAYTON": "DAYTON",
    "DPL": "DPL",
    "DQE": "DUQ",
    "EKPC": "EKPC",
    "JCPL": "JC",
    "METED": "METED",
    "PECO": "PE",
    "PENLC": "PN",
    "PEPCO": "PEP",
    "PL": "PL",
    "PS": "PS",
    "RECO": "RE",
    "UGI": "UGI",
    "VEPCO": "DOM",
    "DUKE": "DEOK"
}

zonal_weather["agg_NodeName"] = zonal_weather["zone"].map(zone_map)

###################### calendar features ##########################

zonal_weather["datetime_ending_ept"] = pd.to_datetime(zonal_weather["datetime_ending_ept"])
zonal_weather["date"] = zonal_weather["datetime_ending_ept"].dt.normalize()
zonal_weather["year"] = zonal_weather["datetime_ending_ept"].dt.year
zonal_weather["month"] = zonal_weather["datetime_ending_ept"].dt.month
zonal_weather["day"] = zonal_weather["datetime_ending_ept"].dt.day
zonal_weather["hour"] = zonal_weather["datetime_ending_ept"].dt.hour
zonal_weather["day_of_week"] = zonal_weather["datetime_ending_ept"].dt.dayofweek
zonal_weather["is_weekend"] = zonal_weather["day_of_week"].isin([5, 6]).astype(int)

us_holidays = holidays.US(observed=True)

zonal_weather["is_holiday"] = zonal_weather["datetime_ending_ept"].apply(
    lambda x: int(x.date() in us_holidays)
)

zonal_weather["WkDayBeforeHol"] = zonal_weather["datetime_ending_ept"].apply(
    lambda x: int((x + pd.Timedelta(days=1)).date() in us_holidays and x.dayofweek < 5)
)

zonal_weather["WkDayAfterHol"] = zonal_weather["datetime_ending_ept"].apply(
    lambda x: int((x - pd.Timedelta(days=1)).date() in us_holidays and x.dayofweek < 5)
)

def build_event_calendar(start_year, end_year):
    rows = []

    us = holidays.US(years=range(start_year, end_year + 1), observed=True)
    for d, name in us.items():
        rows.append((pd.Timestamp(d), name))

    for y in range(start_year, end_year + 1):
        e = pd.Timestamp(easter(y))
        rows += [
            (e - pd.Timedelta(days=2), "Good Friday"),
            (e, "Easter"),
            (pd.Timestamp(f"{y}-12-24"), "Christmas Eve"),
            (pd.Timestamp(f"{y}-12-31"), "New Year's Eve"),
            (pd.Timestamp(f"{y}-10-31"), "Halloween"),
        ]

        tg = [d for d, n in holidays.US(years=[y]).items() if "Thanksgiving" in n][0]
        rows += [
            (pd.Timestamp(tg) + pd.Timedelta(days=1), "Black Friday"),
            (pd.Timestamp(tg) + pd.Timedelta(days=4), "Cyber Monday"),
        ]

    event_df = pd.DataFrame(rows, columns=["date", "event_name"])
    event_df = (
        event_df.groupby("date")["event_name"]
        .apply(lambda x: ", ".join(sorted(set(x))))
        .reset_index()
        .sort_values("date")
    )
    return event_df

start_year = zonal_weather["year"].min()
end_year = zonal_weather["year"].max()

event_df = build_event_calendar(start_year, end_year)

zonal_weather = zonal_weather.merge(event_df, on="date", how="left")
zonal_weather["is_event"] = zonal_weather["event_name"].notna().astype(int)
zonal_weather.drop(columns=["event_name"], inplace=True)

zonal_wc = zonal_weather[zonal_weather["agg_NodeName"] != "UGI"].copy()

################## load data ###########################

hourly_load = pd.read_csv(
    f"data/raw_pjm_hrl_load_metered/raw_pjm_hrl_load_metered_{today_str}.csv",
    skiprows=[1],
    usecols=lambda c: c not in ["datetime_beginning_utc", "auto_key", "is_verified", "insert_datetime"]
)

hourly_load["MW"] = pd.to_numeric(hourly_load["MW"], errors="coerce")
hourly_load["datetime_beginning_ept"] = pd.to_datetime(hourly_load["datetime_beginning_ept"])
hourly_load["datetime_beginning_ept"] += pd.Timedelta(hours=1)

hourly_load = hourly_load.rename(columns={"datetime_beginning_ept": "datetime_ending_ept"})

hourly_rto_load = hourly_load[hourly_load["zone"] == "RTO"].copy()

hourly_rto_load = (
    hourly_rto_load
    .groupby(["datetime_ending_ept", "zone"], as_index=False)
    .agg({
        "MW": "first",
        "nerc_region": "first",
        "mkt_region": "first",
    })
)

################# append load data ########################

append_load = pd.read_csv(
    f"data/pjm_All_Instantaneous_Load_rt5/pjm_All_Instantaneous_Load_rt5_{today_str}.csv",
    skiprows=[1]
)

append_load["instantaneous_load"] = pd.to_numeric(append_load["instantaneous_load"], errors="coerce")
append_load["datetime_beginning_ept"] = pd.to_datetime(append_load["datetime_beginning_ept"])

append_load = (
    append_load
    .assign(
        datetime_ending_ept=lambda x: x["datetime_beginning_ept"].dt.floor("h") + pd.Timedelta(hours=1)
    )
    .groupby(["area", "datetime_ending_ept"], as_index=False)["instantaneous_load"]
    .mean()
    .rename(columns={"instantaneous_load": "MW"})
)

append_pjm = append_load[append_load["area"] == "PJM RTO"].copy()
max_dt_hourly = hourly_rto_load["datetime_ending_ept"].max() - pd.Timedelta(days=2)

hourly_rto_load = hourly_rto_load.loc[
    hourly_rto_load["datetime_ending_ept"] <= max_dt_hourly,
    ["datetime_ending_ept", "zone", "MW", "nerc_region", "mkt_region"]
].copy()

append_pjm_new = append_pjm.loc[
    append_pjm["datetime_ending_ept"] > max_dt_hourly,
    ["datetime_ending_ept", "MW"]
].copy()

append_pjm_new["zone"] = "RTO"
append_pjm_new["nerc_region"] = "RTO"
append_pjm_new["mkt_region"] = "RTO"

hourly_rto_load = pd.concat(
    [hourly_rto_load, append_pjm_new[["datetime_ending_ept", "zone", "MW", "nerc_region", "mkt_region"]]],
    ignore_index=True
).sort_values("datetime_ending_ept").reset_index(drop=True)


############################# normalization
df = hourly_rto_load.copy()
df["datetime_ending_ept"] = pd.to_datetime(df["datetime_ending_ept"])

# Extract year/month
df["year"] = df["datetime_ending_ept"].dt.year
df["month"] = df["datetime_ending_ept"].dt.month

# Filter March
march = df[df["month"] == 3]

# Median MW per year
march_median = march.groupby("year")["MW"].median().sort_index()

# YoY growth rate
march_growth = march_median.pct_change()

# Optional: convert to dict for later use
growth_dict = march_growth.dropna().to_dict()

df["date"] = df["datetime_ending_ept"].dt.floor("D")
t0 = df["date"].max()

def growth_factor(dt, t0, growth_dict):
    if dt >= t0:
        return 1.0

    factor = 1.0
    current = dt

    while current < t0:
        year_end = pd.Timestamp(year=current.year + 1, month=1, day=1)
        segment_end = min(year_end, t0)

        frac_year = (segment_end - current).days / 365.25

        # Use March-derived growth rate for that year
        r = growth_dict.get(current.year + 1, 0.0)

        factor *= (1 + r) ** frac_year
        current = segment_end

    return factor

df["growth_factor"] = df["date"].apply(lambda d: growth_factor(d, t0, growth_dict))

# Normalize to latest-year equivalent
df["MW_normalized"] = df["MW"] * df["growth_factor"]

hourly_rto_load = df[
    ["datetime_ending_ept", "zone", "MW_normalized"]
].rename(columns={"MW_normalized": "MW"})

############################################### RTO prediction #####################################################

weather_cols = weather_vars.copy()
other_cols = [c for c in zonal_wc.columns if c not in weather_cols + ["agg_NodeName"]]

agg_dict = {col: ["median", "std"] for col in weather_cols}
agg_dict.update({col: "first" for col in other_cols if col != "datetime_ending_ept"})

rto_weather = (
    zonal_wc
    .groupby("datetime_ending_ept", as_index=False)
    .agg(agg_dict)
)

rto_weather.columns = [
    col if isinstance(col, str) else f"{col[0]}_std" if col[1] == "std" else col[0]
    for col in rto_weather.columns
]

################################## alert/warning/action ###########################

alert = pd.read_csv("data/emergencymessages/alerts.csv")
alert["start"] = pd.to_datetime(alert["start"].astype(str).str[:-6])
alert["end"] = pd.to_datetime(alert["end"].astype(str).str[:-6])

warning = pd.read_csv("data/emergencymessages/warnings.csv")
warning["start"] = pd.to_datetime(warning["start"].astype(str).str[:-6])
warning["end"] = pd.to_datetime(warning["end"].astype(str).str[:-6])

action = pd.read_csv("data/emergencymessages/actions.csv")
action["start"] = pd.to_datetime(action["start"].astype(str).str[:-6])
action["end"] = pd.to_datetime(action["end"].astype(str).str[:-6])

rto_weather["datetime_ending_ept"] = pd.to_datetime(rto_weather["datetime_ending_ept"])
rto_weather["is_alert"] = 0
rto_weather["is_warning"] = 0
rto_weather["is_action"] = 0

for _, row in alert.iterrows():
    mask = rto_weather["datetime_ending_ept"].between(row["start"], row["end"])
    rto_weather.loc[mask, "is_alert"] = 1

for _, row in warning.iterrows():
    mask = rto_weather["datetime_ending_ept"].between(row["start"], row["end"])
    rto_weather.loc[mask, "is_warning"] = 1

for _, row in action.iterrows():
    mask = rto_weather["datetime_ending_ept"].between(row["start"], row["end"])
    rto_weather.loc[mask, "is_action"] = 1

########################### temp ##################################################

daily_temp = (
    rto_weather.groupby("date", as_index=False)["temperature_2m"]
    .agg(temp_min="min", temp_max="max")
)

rto_weather = rto_weather.merge(daily_temp, on="date", how="left")

rto_weather["temp_f"] = rto_weather["temperature_2m"] * 9 / 5 + 32

base_temp = 65
rto_weather["HDD"] = np.maximum(base_temp - rto_weather["temp_f"], 0)
rto_weather["CDD"] = np.maximum(rto_weather["temp_f"] - base_temp, 0)

rto_weather["HDD_wind"] = rto_weather["HDD"] * rto_weather["wind_speed_10m"]
rto_weather["CDD_cloud"] = rto_weather["CDD"] * (1 - rto_weather["cloud_cover"] / 100)

rto_weather["wind_dir_10m_sin"] = np.sin(np.deg2rad(rto_weather["wind_direction_10m"]))
rto_weather["wind_dir_10m_cos"] = np.cos(np.deg2rad(rto_weather["wind_direction_10m"]))

RH = rto_weather["relative_humidity_2m"]
T = rto_weather["temp_f"]

wind_chill = (
    35.74
    + 0.6215 * T
    - 35.75 * (rto_weather["wind_speed_10m"] ** 0.16)
    + 0.4275 * T * (rto_weather["wind_speed_10m"] ** 0.16)
)

heat_index_f = (
    -42.379
    + 2.04901523 * T
    + 10.14333127 * RH
    - 0.22475541 * T * RH
    - 0.00683783 * T**2
    - 0.05481717 * RH**2
    + 0.00122874 * T**2 * RH
    + 0.00085282 * T * RH**2
    - 0.00000199 * T**2 * RH**2
)

rto_weather["feels_like_temp"] = np.where(
    (rto_weather["temp_f"] <= 50) & (rto_weather["wind_speed_10m"] > 3),
    wind_chill,
    np.where(
        (rto_weather["temp_f"] >= 80) & (RH >= 40),
        heat_index_f,
        rto_weather["temp_f"]
    )
)

temp_diff_1h_abs = rto_weather["temperature_2m"].diff().abs()
rto_weather["temp_total_variation_6h"] = temp_diff_1h_abs.rolling(6).sum()
rto_weather["temp_total_variation_12h"] = temp_diff_1h_abs.rolling(12).sum()
rto_weather["temp_total_variation_24h"] = temp_diff_1h_abs.rolling(24).sum()

rto_weather["temp_fcst_1h"] = rto_weather["temperature_2m"].shift(-1)

horizon = 6
rto_weather["temp_lead_mean_6h"] = sum(
    rto_weather["temperature_2m"].shift(-i)
    for i in range(1, horizon + 1)
) / horizon

rto_weather.dropna(inplace=True)
rto_weather.drop(columns=["zone"], inplace=True, errors="ignore")

# merge
rto_load = pd.merge(
    hourly_rto_load,
    rto_weather,
    on="datetime_ending_ept",
    how="right"
)

rto_load = rto_load.sort_values("datetime_ending_ept").copy()
rto_load["datetime_ending_ept"] = pd.to_datetime(rto_load["datetime_ending_ept"])

################################# lag #########################################

rto_load["hour"] = rto_load["datetime_ending_ept"].dt.hour
rto_load["day_shift"] = np.where(rto_load["hour"].isin(range(1, 10)), 1, 2)

hist = (
    rto_load[["datetime_ending_ept", "MW"]]
    .drop_duplicates("datetime_ending_ept")
    .sort_values("datetime_ending_ept")
    .set_index("datetime_ending_ept")
)

lag_hours = [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168]

helper_cols = []
feature_cols = []

for lag in lag_hours:
    time_col = f"lag{lag}_time"
    feat_col = f"MW_lag_{lag}"

    rto_load[time_col] = (
        rto_load["datetime_ending_ept"]
        - pd.to_timedelta(rto_load["day_shift"], unit="D")
        - pd.Timedelta(hours=lag)
    )

    rto_load[feat_col] = hist["MW"].reindex(rto_load[time_col]).to_numpy()

    helper_cols.append(time_col)
    feature_cols.append(feat_col)

for window in [24, 48, 72, 168]:
    hist[f"MW_roll_{window}_base"] = hist["MW"].shift(1).rolling(window).mean()

for window in [24, 168]:
    hist[f"MW_std_{window}_base"] = hist["MW"].shift(1).rolling(window).std()

rto_load["roll_time"] = (
    rto_load["datetime_ending_ept"]
    - pd.to_timedelta(rto_load["day_shift"], unit="D")
)

for window in [24, 48, 72, 168]:
    rto_load[f"MW_roll_{window}"] = hist[f"MW_roll_{window}_base"].reindex(rto_load["roll_time"]).to_numpy()

for window in [24, 168]:
    rto_load[f"MW_std_{window}"] = hist[f"MW_std_{window}_base"].reindex(rto_load["roll_time"]).to_numpy()

feature_cols += [
    "MW_roll_24", "MW_roll_48", "MW_roll_72", "MW_roll_168",
    "MW_std_24", "MW_std_168"
]

rto_load = rto_load.dropna(subset=feature_cols).copy()
rto_load.drop(columns=["day_shift", "roll_time"] + helper_cols, inplace=True)

######################### model ##############################

results = []
feat_imp_results = []

start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=offset)
end_date = start_date + pd.Timedelta(days=2) - pd.Timedelta(hours=1)

current_date = start_date

while current_date <= end_date:

    print(f"Running for {current_date.date()}")

    cutoff = (current_date - pd.Timedelta(days=1)).replace(hour=10)

    train_df = rto_load[rto_load["datetime_ending_ept"] <= cutoff].copy()

    test_start = current_date.replace(hour=1)
    test_end = current_date.replace(hour=23) + pd.Timedelta(hours=1)

    test_df = rto_load[
        (rto_load["datetime_ending_ept"] >= test_start) &
        (rto_load["datetime_ending_ept"] <= test_end)
    ].copy()

    if train_df.empty or test_df.empty:
        print("Skipping because train/test is empty")
        current_date += pd.Timedelta(days=1)
        continue

    target = "MW"
    drop_cols = ["MW", "datetime_ending_ept", "zone", "date"]
    features = [col for col in train_df.columns if col not in drop_cols]

    valid_hours = 24 * 14
    train_part = train_df.iloc[:-valid_hours].copy()
    valid_part = train_df.iloc[-valid_hours:].copy()

    X_train = train_part[features]
    y_train = train_part[target]
    X_valid = valid_part[features]
    y_valid = valid_part[target]
    X_test = test_df[features]

    model = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

    feat_imp = pd.DataFrame({
        "date": current_date,
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feat_imp["rank"] = range(1, len(feat_imp) + 1)
    feat_imp_results.append(feat_imp)

    test_df["MW_pred"] = model.predict(X_test)

    forecast_df = (
        rto_load_forecast[
            (rto_load_forecast["forecast_datetime_ending_ept"] >= test_start) &
            (rto_load_forecast["forecast_datetime_ending_ept"] <= test_end)
        ]
        .sort_values(["forecast_datetime_ending_ept"])
        .copy()
    )

    forecast_df = forecast_df.rename(columns={
        "forecast_datetime_ending_ept": "datetime_ending_ept"
    })

    forecast_df["datetime_ending_ept"] = pd.to_datetime(forecast_df["datetime_ending_ept"])

    compare_df = pd.merge(
        test_df,
        forecast_df[["datetime_ending_ept", "forecast_load_mw"]],
        on="datetime_ending_ept",
        how="inner"
    )

    mae = mean_absolute_error(compare_df["forecast_load_mw"], compare_df["MW_pred"])
    rmse = np.sqrt(mean_squared_error(compare_df["forecast_load_mw"], compare_df["MW_pred"]))

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    compare_df["date"] = current_date
    compare_df["MAE"] = mae
    compare_df["RMSE"] = rmse

    results.append(compare_df)

    current_date += pd.Timedelta(days=1)

# combine all days
final_results = pd.concat(results, ignore_index=True)

start_str = today.strftime("%m%d")
end_str = (today + timedelta(days=1)).strftime("%m%d")
run_str = today.strftime("%y%m%d")

final_results[
    ["datetime_ending_ept", "date", "hour", "MW_pred", "forecast_load_mw"]
].to_csv(
    f"data/prediction/RTO_prediction_{start_str}_to_{end_str}_at_{run_str}_try.csv",
    index=False
)
