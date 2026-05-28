import pandas as pd
import numpy as np
import lightgbm as lgb

from datetime import datetime, timedelta
import holidays

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dateutil.easter import easter

############################ parameters ##########################################

offset = 0
today = datetime.today() - timedelta(days=offset)
today_str = today.strftime("%m%d")
run_str = today.strftime("%y%m%d")

############################ helper functions ##########################################

WEATHER_VARS = [
    "temperature_2m", "apparent_temperature", "dew_point_2m",
    "relative_humidity_2m", "precipitation", "rain", "snowfall", "snow_depth",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "surface_pressure", "pressure_msl", "et0_fao_evapotranspiration",
    "vapour_pressure_deficit", "shortwave_radiation", "direct_radiation",
    "diffuse_radiation", "global_tilted_irradiance",
    "direct_normal_irradiance", "terrestrial_radiation"
]

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

def apply_time_flags(df):
    df = df.copy()
    df["datetime_ending_ept"] = pd.to_datetime(df["datetime_ending_ept"])
    df["date"] = df["datetime_ending_ept"].dt.normalize()
    df["year"] = df["datetime_ending_ept"].dt.year
    df["month"] = df["datetime_ending_ept"].dt.month
    df["day"] = df["datetime_ending_ept"].dt.day
    df["hour"] = df["datetime_ending_ept"].dt.hour
    df["day_of_week"] = df["datetime_ending_ept"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    us_holidays = holidays.US(observed=True)
    df["is_holiday"] = df["datetime_ending_ept"].apply(lambda x: int(x.date() in us_holidays))
    df["WkDayBeforeHol"] = df["datetime_ending_ept"].apply(
        lambda x: int((x + pd.Timedelta(days=1)) in us_holidays and x.dayofweek < 5)
    )
    df["WkDayAfterHol"] = df["datetime_ending_ept"].apply(
        lambda x: int((x - pd.Timedelta(days=1)) in us_holidays and x.dayofweek < 5)
    )
    return df

def add_interval_flag(df, intervals, col_name):
    df = df.copy()
    df[col_name] = 0
    for _, row in intervals.iterrows():
        mask = (
            (df["datetime_ending_ept"] >= row["start"]) &
            (df["datetime_ending_ept"] <= row["end"])
        )
        df.loc[mask, col_name] = 1
    return df

def add_common_features(df):
    df = df.copy()

    daily_temp = (
        df.groupby("date", as_index=False)["temperature_2m"]
        .agg(temp_min="min", temp_max="max")
    )
    df = df.merge(daily_temp, on="date", how="left")

    df["temp_f"] = df["temperature_2m"] * 9 / 5 + 32
    base_temp = 65

    df["HDD"] = np.maximum(base_temp - df["temp_f"], 0)
    df["CDD"] = np.maximum(df["temp_f"] - base_temp, 0)
    df["HDD_wind"] = df["HDD"] * df["wind_speed_10m"]
    df["CDD_cloud"] = df["CDD"] * (1 - df["cloud_cover"] / 100)

    df["wind_dir_10m_sin"] = np.sin(np.deg2rad(df["wind_direction_10m"]))
    df["wind_dir_10m_cos"] = np.cos(np.deg2rad(df["wind_direction_10m"]))

    wind_chill = (
        35.74
        + 0.6215 * df["temp_f"]
        - 35.75 * (df["wind_speed_10m"] ** 0.16)
        + 0.4275 * df["temp_f"] * (df["wind_speed_10m"] ** 0.16)
    )

    RH = df["relative_humidity_2m"]
    T = df["temp_f"]

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

    df["feels_like_temp"] = np.where(
        (df["temp_f"] <= 50) & (df["wind_speed_10m"] > 3),
        wind_chill,
        np.where((df["temp_f"] >= 80) & (RH >= 40), heat_index_f, df["temp_f"])
    )

    df["temp_diff_1h_abs"] = df["temperature_2m"].diff().abs()
    df["temp_total_variation_6h"] = df["temp_diff_1h_abs"].rolling(6).sum()
    df["temp_total_variation_12h"] = df["temp_diff_1h_abs"].rolling(12).sum()
    df["temp_total_variation_24h"] = df["temp_diff_1h_abs"].rolling(24).sum()

    return df.drop(columns=["temp_diff_1h_abs"], errors="ignore")

def add_lag_features(df, target_col, lag_hours=None):
    if lag_hours is None:
        lag_hours = [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168]

    df = df.sort_values("datetime_ending_ept").copy()
    df["hour"] = df["datetime_ending_ept"].dt.hour
    df["day_shift"] = np.where(df["hour"].isin(range(1, 10)), 1, 2)

    hist = (
        df[["datetime_ending_ept", target_col]]
        .drop_duplicates("datetime_ending_ept")
        .sort_values("datetime_ending_ept")
        .set_index("datetime_ending_ept")
    )

    helper_cols = []
    feature_cols = []

    for lag in lag_hours:
        time_col = f"lag{lag}_time"
        feat_col = f"{target_col}_lag_{lag}"

        df[time_col] = (
            df["datetime_ending_ept"]
            - pd.to_timedelta(df["day_shift"], unit="D")
            - pd.Timedelta(hours=lag)
        )
        df[feat_col] = hist[target_col].reindex(df[time_col]).to_numpy()

        helper_cols.append(time_col)
        feature_cols.append(feat_col)

    hist["roll_24"] = hist[target_col].shift(1).rolling(24).mean()
    hist["roll_48"] = hist[target_col].shift(1).rolling(48).mean()
    hist["roll_72"] = hist[target_col].shift(1).rolling(72).mean()
    hist["roll_168"] = hist[target_col].shift(1).rolling(168).mean()
    hist["std_24"] = hist[target_col].shift(1).rolling(24).std()
    hist["std_168"] = hist[target_col].shift(1).rolling(168).std()

    df["roll_time"] = df["datetime_ending_ept"] - pd.to_timedelta(df["day_shift"], unit="D")
    df[f"{target_col}_roll_24"] = hist["roll_24"].reindex(df["roll_time"]).to_numpy()
    df[f"{target_col}_roll_48"] = hist["roll_48"].reindex(df["roll_time"]).to_numpy()
    df[f"{target_col}_roll_72"] = hist["roll_72"].reindex(df["roll_time"]).to_numpy()
    df[f"{target_col}_roll_168"] = hist["roll_168"].reindex(df["roll_time"]).to_numpy()
    df[f"{target_col}_std_24"] = hist["std_24"].reindex(df["roll_time"]).to_numpy()
    df[f"{target_col}_std_168"] = hist["std_168"].reindex(df["roll_time"]).to_numpy()

    feature_cols += [
        f"{target_col}_roll_24", f"{target_col}_roll_48",
        f"{target_col}_roll_72", f"{target_col}_roll_168",
        f"{target_col}_std_24", f"{target_col}_std_168",
    ]

    df = df.dropna(subset=feature_cols).copy()
    df = df.drop(columns=["day_shift", "roll_time"] + helper_cols, errors="ignore")
    return df

def prepare_sector_dataset(df, target_col, alert_df, warning_df, action_df):
    df = df.copy()
    df["datetime_ending_ept"] = pd.to_datetime(df["datetime_ending_ept"])
    df = add_interval_flag(df, alert_df, "is_alert")
    df = add_interval_flag(df, warning_df, "is_warning")
    df = add_interval_flag(df, action_df, "is_action")
    df = add_common_features(df)
    df = add_lag_features(df, target_col)
    return df

def run_sector_model(
    df,
    target_col,
    pred_col,
    forecast_col=None,
    weighted_eval=False,
    offset=0
):
    results = []
    feat_imp_results = []

    start_date = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize()
    end_date = start_date + pd.Timedelta(days=2) - pd.Timedelta(hours=1)
    current_date = start_date

    while current_date <= end_date:
        print(f"Running for {target_col} on {current_date.date()}")

        cutoff = (current_date - pd.Timedelta(days=1)).replace(hour=10)

        train_df = df[df["datetime_ending_ept"] <= cutoff].copy()

        test_start = current_date.replace(hour=1)
        test_end = current_date.replace(hour=23) + pd.Timedelta(hours=1)

        test_df = df[
            (df["datetime_ending_ept"] >= test_start) &
            (df["datetime_ending_ept"] <= test_end)
        ].copy()

        drop_cols = [target_col, "datetime_ending_ept", "zone", "date"]
        if forecast_col is not None and forecast_col in train_df.columns:
            drop_cols.append(forecast_col)

        features = [c for c in train_df.columns if c not in drop_cols]

        valid_hours = 24 * 14
        train_part = train_df.iloc[:-valid_hours].copy()
        valid_part = train_df.iloc[-valid_hours:].copy()

        X_train = train_part[features]
        y_train = train_part[target_col]
        X_valid = valid_part[features]
        y_valid = valid_part[target_col]
        X_test = test_df[features]

        model = LGBMRegressor(
            objective="regression",
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

        if weighted_eval and forecast_col is not None:
            train_dev_pct = (
                (train_part[target_col] - train_part[forecast_col]).abs()
                / train_part[forecast_col].abs().clip(lower=1e-6)
            )
            valid_dev_pct = (
                (valid_part[target_col] - valid_part[forecast_col]).abs()
                / valid_part[forecast_col].abs().clip(lower=1e-6)
            )

            train_weights = np.where(train_dev_pct > 0.05, 2, 1)
            valid_weights = np.where(valid_dev_pct > 0.05, 2, 1)
            valid_forecast = valid_part[forecast_col].to_numpy()

            def extreme_rmse_eval(y_true, y_pred):
                extreme_mask = (
                    np.abs(y_true - valid_forecast) /
                    np.clip(np.abs(valid_forecast), 1e-6, None)
                ) > 0.05

                if extreme_mask.sum() == 0:
                    return ("extreme_rmse", 0.0, False)

                rmse = np.sqrt(np.mean((y_true[extreme_mask] - y_pred[extreme_mask]) ** 2))
                return ("extreme_rmse", rmse, False)

            model.fit(
                X_train,
                y_train,
                sample_weight=train_weights,
                eval_set=[(X_valid, y_valid)],
                eval_sample_weight=[valid_weights],
                eval_metric=lambda yt, yp: [extreme_rmse_eval(yt, yp)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
            )
        else:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
            )

        feat_imp = pd.DataFrame({
            "date": current_date,
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        feat_imp["rank"] = range(1, len(feat_imp) + 1)
        feat_imp_results.append(feat_imp)

        test_df[pred_col] = model.predict(X_test)

        if forecast_col is not None and forecast_col in test_df.columns:
            mae = mean_absolute_error(test_df[forecast_col], test_df[pred_col])
            rmse = np.sqrt(mean_squared_error(test_df[forecast_col], test_df[pred_col]))
            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            test_df["MAE"] = mae
            test_df["RMSE"] = rmse

        test_df["date"] = current_date
        results.append(test_df)

        current_date += pd.Timedelta(days=1)

    final_df = pd.concat(results).reset_index(drop=True)
    feat_imp_df = pd.concat(feat_imp_results).reset_index(drop=True)
    return final_df, feat_imp_df

############################ load forecast data ##########################################

load_forecast = pd.read_csv(
    f"data/load_frcstd_7_day/load_frcstd_7_day_{today_str}.csv",
    skiprows=[1]
)

load_forecast["forecast_load_mw"] = pd.to_numeric(load_forecast["forecast_load_mw"], errors="coerce")
load_forecast = load_forecast[
    ["evaluated_at_datetime_ept", "forecast_datetime_ending_ept", "forecast_area", "forecast_load_mw"]
].copy()

load_forecast["forecast_datetime_ending_ept"] = pd.to_datetime(load_forecast["forecast_datetime_ending_ept"])

forecast_to_agg = {
    "AE/MIDATL": "AE", "AEP": "AEP", "AP": "AP", "ATSI": "ATSI",
    "BG&E/MIDATL": "BGE", "COMED": "COMED", "DAYTON": "DAYTON",
    "DEOK": "DEOK", "DOMINION": "DOM", "DP&L/MIDATL": "DPL",
    "DUQUESNE": "DUQ", "EKPC": "EKPC", "JCP&L/MIDATL": "JC",
    "METED/MIDATL": "METED", "PECO/MIDATL": "PE", "PENELEC/MIDATL": "PN",
    "PEPCO/MIDATL": "PEP", "PPL/MIDATL": "PL", "PSE&G/MIDATL": "PS",
    "RECO/MIDATL": "RE", "UGI/MIDATL": "UGI",
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
rto_load_forecast["hour"] = rto_load_forecast["forecast_datetime_ending_ept"].dt.hour
rto_load_forecast = rto_load_forecast.rename(columns={"agg_NodeName": "zone"})

############################### weather data ##########################################

hist_files = [
    "data/weather/weather_hist_2023.csv",
    "data/weather/weather_hist_2024.csv",
    "data/weather/weather_hist_2025.csv",
    "data/weather/weather_hist_2026.csv",
]
forecast_file = f"data/weather/weather_forecast_{run_str}.csv"

hist_weather = pd.concat([pd.read_csv(f) for f in hist_files], ignore_index=True)
hist_weather = hist_weather.drop(columns=["weather_code"], errors="ignore")
hist_weather = hist_weather.rename(columns={"time": "datetime_ending_ept"})
hist_weather["datetime_ending_ept"] = pd.to_datetime(hist_weather["datetime_ending_ept"])

forecast_weather = pd.read_csv(forecast_file)
forecast_weather = forecast_weather.drop(columns=["weather_code"], errors="ignore")
forecast_weather = forecast_weather.rename(columns={"time": "datetime_ending_ept"})
forecast_weather["datetime_ending_ept"] = pd.to_datetime(forecast_weather["datetime_ending_ept"])

run_dt = pd.to_datetime(run_str, format="%y%m%d")

hist_weather = hist_weather[hist_weather["datetime_ending_ept"] <= run_dt].copy()
forecast_weather = forecast_weather[forecast_weather["datetime_ending_ept"] > run_dt].copy()

dedupe_cols = ["zone", "weather_station", "lat", "lon", "datetime_ending_ept"]
hist_weather = hist_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
forecast_weather = forecast_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

forecast_weather["hour"] = forecast_weather["datetime_ending_ept"].dt.hour

hourly_avg = (
    forecast_weather.groupby(["zone", "weather_station", "hour"])[WEATHER_VARS]
    .mean()
    .reset_index()
)

forecast_weather = forecast_weather.merge(
    hourly_avg,
    on=["zone", "weather_station", "hour"],
    how="left",
    suffixes=("", "_hour_avg")
)

for col in WEATHER_VARS:
    forecast_weather[col] = forecast_weather[col].fillna(forecast_weather[f"{col}_hour_avg"])

forecast_weather = forecast_weather.drop(
    columns=["hour"] + [f"{c}_hour_avg" for c in WEATHER_VARS],
    errors="ignore"
)

zonal_weather = pd.concat([hist_weather, forecast_weather], ignore_index=True)
zonal_weather = zonal_weather.sort_values(["zone", "weather_station", "datetime_ending_ept"]).reset_index(drop=True)

for var in WEATHER_VARS:
    zonal_weather[f"{var}_weighted"] = zonal_weather[var] * zonal_weather["weather_weight"]

zonal_weather = (
    zonal_weather.groupby(["datetime_ending_ept", "zone"])[[f"{v}_weighted" for v in WEATHER_VARS]]
    .sum()
    .reset_index()
)

zonal_weather = zonal_weather.rename(columns={f"{v}_weighted": v for v in WEATHER_VARS})

zone_map = {
    "AE": "AE", "AEP": "AEP", "APS": "AP", "ATSI": "ATSI", "BGE": "BGE",
    "COMED": "COMED", "DAYTON": "DAYTON", "DPL": "DPL", "DQE": "DUQ",
    "EKPC": "EKPC", "JCPL": "JC", "METED": "METED", "PECO": "PE",
    "PENLC": "PN", "PEPCO": "PEP", "PL": "PL", "PS": "PS", "RECO": "RE",
    "UGI": "UGI", "VEPCO": "DOM", "DUKE": "DEOK"
}

zonal_weather["agg_NodeName"] = zonal_weather["zone"].map(zone_map)
zonal_weather = apply_time_flags(zonal_weather)

start_year = zonal_weather["year"].min()
end_year = zonal_weather["year"].max()
event_df = build_event_calendar(start_year, end_year)

zonal_weather = zonal_weather.merge(event_df, on="date", how="left")
zonal_weather["is_event"] = zonal_weather["event_name"].notna().astype(int)
zonal_weather = zonal_weather.drop(columns=["event_name"], errors="ignore")

zonal_wc = zonal_weather[zonal_weather["agg_NodeName"] != "UGI"].copy()

########################### load data ##########################################

hourly_load = pd.read_csv(
    f"data/raw_pjm_hrl_load_metered/raw_pjm_hrl_load_metered_{today_str}.csv",
    skiprows=[1],
    usecols=lambda c: c not in ["datetime_beginning_utc", "auto_key", "is_verified", "insert_datetime"]
)

hourly_load["MW"] = pd.to_numeric(hourly_load["MW"], errors="coerce")
hourly_load["datetime_beginning_ept"] = pd.to_datetime(hourly_load["datetime_beginning_ept"])
hourly_load["datetime_beginning_ept"] += pd.Timedelta(hours=1)

hourly_load = hourly_load.rename(columns={"datetime_beginning_ept": "datetime_ending_ept"})
hourly_load = hourly_load[["datetime_ending_ept", "zone"] + [c for c in hourly_load.columns if c not in ["datetime_ending_ept", "zone"]]]

hourly_rto_load = hourly_load[hourly_load["zone"] == "RTO"].copy()
hourly_rto_load = (
    hourly_rto_load.groupby(["datetime_ending_ept", "zone"], as_index=False)
    .agg({"MW": "first", "nerc_region": "first", "mkt_region": "first"})
)

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
    .rename(columns={"instantaneous_load": "load_mw_hourly_avg"})
)

append_pjm = append_load.loc[append_load["area"] == "PJM RTO"].copy()

max_dt_hourly = hourly_rto_load["datetime_ending_ept"].max() - pd.Timedelta(days=2)

hourly_rto_load = hourly_rto_load.loc[
    hourly_rto_load["datetime_ending_ept"] <= max_dt_hourly,
    ["datetime_ending_ept", "zone", "MW", "nerc_region", "mkt_region"]
].copy()

append_pjm_new = append_pjm.loc[
    append_pjm["datetime_ending_ept"] > max_dt_hourly,
    ["datetime_ending_ept", "load_mw_hourly_avg"]
].copy()

append_pjm_new = append_pjm_new.rename(columns={"load_mw_hourly_avg": "MW"})
append_pjm_new["zone"] = "RTO"
append_pjm_new["nerc_region"] = "RTO"
append_pjm_new["mkt_region"] = "RTO"

hourly_rto_load = pd.concat(
    [hourly_rto_load, append_pjm_new[["datetime_ending_ept", "zone", "MW", "nerc_region", "mkt_region"]]],
    ignore_index=True
).sort_values("datetime_ending_ept").reset_index(drop=True)

weather_cols = WEATHER_VARS
other_cols = [c for c in zonal_wc.columns if c not in weather_cols + ["agg_NodeName"]]

agg_dict = {col: ["median", "std"] for col in weather_cols}
agg_dict.update({col: "first" for col in other_cols if col != "datetime_ending_ept"})

rto_weather = zonal_wc.groupby("datetime_ending_ept", as_index=False).agg(agg_dict)
rto_weather.columns = [
    col if isinstance(col, str) else (f"{col[0]}_std" if col[1] == "std" else col[0])
    for col in rto_weather.columns
]
rto_weather = rto_weather.drop(columns=["zone"], errors="ignore")

rto_load = pd.merge(hourly_rto_load, rto_weather, on="datetime_ending_ept", how="right")
rto_load = rto_load.drop(columns=["mkt_region", "nerc_region"], errors="ignore").sort_values("datetime_ending_ept")
rto_load["datetime_ending_ept"] = pd.to_datetime(rto_load["datetime_ending_ept"])
rto_load["hour"] = rto_load["datetime_ending_ept"].dt.hour

################################### RTO sector split ##################################

hourly_sector_weight = pd.read_csv("data/hrl_sct_wt.csv")
rto_sector_shares = pd.DataFrame([
    {"zone": "RTO", "residential_share": 0.37, "commercial_share": 0.37, "industrial_share": 0.25}
])

def split_sector_load(df, mw_col, output_cols):
    out = df.copy()
    out = out.merge(hourly_sector_weight, on="hour", how="left")
    out = out.merge(rto_sector_shares, on="zone", how="left")

    out["wtd_res"] = out["residential_share"] * out["w_res"]
    out["wtd_com"] = out["commercial_share"] * out["w_com"]
    out["wtd_ind"] = out["industrial_share"] * out["w_ind"]

    weight_sum = out[["wtd_res", "wtd_com", "wtd_ind"]].sum(axis=1)

    out[output_cols[0]] = out[mw_col] * out["wtd_res"] / weight_sum
    out[output_cols[1]] = out[mw_col] * out["wtd_com"] / weight_sum
    out[output_cols[2]] = out[mw_col] * out["wtd_ind"] / weight_sum
    return out

rto_load_forecast = split_sector_load(
    rto_load_forecast,
    "forecast_load_mw",
    ["res_forecast_load_mw", "com_forecast_load_mw", "ind_forecast_load_mw"]
)

rto_load = split_sector_load(
    rto_load,
    "MW",
    ["res_MW", "com_MW", "ind_MW"]
)

rto_load_forecast = rto_load_forecast.rename(columns={"forecast_datetime_ending_ept": "datetime_ending_ept"})
rto_load_forecast["datetime_ending_ept"] = pd.to_datetime(rto_load_forecast["datetime_ending_ept"])

base_cols = [
    "datetime_ending_ept", "zone",
    *WEATHER_VARS,
    *[f"{c}_std" for c in WEATHER_VARS],
    "date", "year", "month", "day", "hour", "day_of_week",
    "is_weekend", "is_holiday", "WkDayBeforeHol", "WkDayAfterHol", "is_event"
]

res_rto_load = rto_load[base_cols + ["res_MW"]].copy()
com_rto_load = rto_load[base_cols + ["com_MW"]].copy()
ind_rto_load = rto_load[base_cols + ["ind_MW"]].copy()

############################ event intervals ##########################################


alert = pd.read_csv("data/emergencymessages/alerts.csv")
alert["start"] = pd.to_datetime(alert["start"].astype(str).str[:-6])
alert["end"]   = pd.to_datetime(alert["end"].astype(str).str[:-6])

warning = pd.read_csv("data/emergencymessages/warnings.csv")
warning["start"] = pd.to_datetime(warning["start"].astype(str).str[:-6])
warning["end"] = pd.to_datetime(warning["end"].astype(str).str[:-6])

action = pd.read_csv("data/emergencymessages/actions.csv")
action["start"] = pd.to_datetime(action["start"].astype(str).str[:-6])
action["end"]   = pd.to_datetime(action["end"].astype(str).str[:-6])

################################## residential ########################################

res_rto_load = prepare_sector_dataset(res_rto_load, "res_MW", alert, warning, action)
res_rto_load = res_rto_load.merge(
    rto_load_forecast[["datetime_ending_ept", "res_forecast_load_mw"]],
    on="datetime_ending_ept",
    how="left"
)

res_final, res_feat_imp = run_sector_model(
    res_rto_load,
    target_col="res_MW",
    pred_col="res_MW_pred",
    forecast_col="res_forecast_load_mw",
    weighted_eval=True,
    offset=offset
)

################################## commercial ########################################

com_rto_load = prepare_sector_dataset(com_rto_load, "com_MW", alert, warning, action)
com_rto_load = com_rto_load.merge(
    rto_load_forecast[["datetime_ending_ept", "com_forecast_load_mw"]],
    on="datetime_ending_ept",
    how="left"
)

com_final, com_feat_imp = run_sector_model(
    com_rto_load,
    target_col="com_MW",
    pred_col="com_MW_pred",
    forecast_col="com_forecast_load_mw",
    weighted_eval=False,
    offset=offset
)

################################## industrial ########################################

ind_rto_load = prepare_sector_dataset(ind_rto_load, "ind_MW", alert, warning, action)
ind_rto_load = ind_rto_load.merge(
    rto_load_forecast[["datetime_ending_ept", "ind_forecast_load_mw"]],
    on="datetime_ending_ept",
    how="left"
)

ind_final, ind_feat_imp = run_sector_model(
    ind_rto_load,
    target_col="ind_MW",
    pred_col="ind_MW_pred",
    forecast_col="ind_forecast_load_mw",
    weighted_eval=False,
    offset=offset
)

################################## sector union ########################################

pred_df = (
    res_final[["datetime_ending_ept", "res_MW_pred"]]
    .merge(com_final[["datetime_ending_ept", "com_MW_pred"]], on="datetime_ending_ept", how="outer")
    .merge(ind_final[["datetime_ending_ept", "ind_MW_pred"]], on="datetime_ending_ept", how="outer")
)

pred_df["total_MW_pred"] = (
    pred_df["res_MW_pred"] +
    pred_df["com_MW_pred"] +
    pred_df["ind_MW_pred"]
)

pred_df = pred_df.merge(
    rto_load[["datetime_ending_ept", "MW"]],
    on="datetime_ending_ept",
    how="left"
)

pred_df = pred_df.merge(
    rto_load_forecast[["datetime_ending_ept", "forecast_load_mw"]],
    on="datetime_ending_ept",
    how="left"
)

pred_df["date"] = pred_df["datetime_ending_ept"].dt.date
pred_df["hour"] = pred_df["datetime_ending_ept"].dt.hour

daily_metrics = (
    pred_df.groupby("date")
    .apply(lambda df: pd.Series({
        "mae": mean_absolute_error(df["forecast_load_mw"], df["total_MW_pred"]),
        "rmse": np.sqrt(mean_squared_error(df["forecast_load_mw"], df["total_MW_pred"]))
    }))
    .reset_index()
)

start_str = today.strftime("%m%d")
end_str = (today + timedelta(days=1)).strftime("%m%d")

pred_df[
    ["datetime_ending_ept", "date", "hour", "total_MW_pred", "forecast_load_mw"]
].to_csv(
    f"data/prediction/RTO_sector_prediction_{start_str}_to_{end_str}_at_{run_str}_try.csv",
    index=False
)