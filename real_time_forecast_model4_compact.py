import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import holidays
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dateutil.easter import easter

####################################### config ##################################
offset = 0
today = datetime.today() - timedelta(days=offset)
today_str = today.strftime("%m%d")
run_str = today.strftime("%y%m%d")

####################################### helper functions ##################################

def apply_interval_flag(df, intervals_df, flag_col, time_col="datetime_ending_ept"):
    df = df.copy()
    df[flag_col] = 0
    for _, row in intervals_df.iterrows():
        mask = (
            (df[time_col] >= row["start"]) &
            (df[time_col] <= row["end"])
        )
        df.loc[mask, flag_col] = 1
    return df


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


def engineer_common_weather_features(df):
    df = df.copy()
    df["datetime_ending_ept"] = pd.to_datetime(df["datetime_ending_ept"])
    df = df.sort_values(["zone", "datetime_ending_ept"]).copy()

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

    df["wind_chill"] = (
        35.74
        + 0.6215 * df["temp_f"]
        - 35.75 * (df["wind_speed_10m"] ** 0.16)
        + 0.4275 * df["temp_f"] * (df["wind_speed_10m"] ** 0.16)
    )

    RH = df["relative_humidity_2m"]
    T = df["temp_f"]

    df["heat_index_f"] = (
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
        df["wind_chill"],
        np.where(
            (df["temp_f"] >= 80) & (RH >= 40),
            df["heat_index_f"],
            df["temp_f"]
        )
    )

    df["temp_diff_1h_abs"] = df.groupby("zone")["temperature_2m"].diff().abs()
    df["temp_total_variation_6h"] = df.groupby("zone")["temp_diff_1h_abs"].transform(lambda s: s.rolling(6).sum())
    df["temp_total_variation_12h"] = df.groupby("zone")["temp_diff_1h_abs"].transform(lambda s: s.rolling(12).sum())
    df["temp_total_variation_24h"] = df.groupby("zone")["temp_diff_1h_abs"].transform(lambda s: s.rolling(24).sum())

    df["temp_fcst_1h"] = df.groupby("zone")["temperature_2m"].shift(-1)

    horizon = 6
    df["temp_lead_mean_6h"] = (
        sum(df.groupby("zone")["temperature_2m"].shift(-i) for i in range(1, horizon + 1)) / horizon
    )

    df.drop(columns=["wind_chill", "temp_diff_1h_abs", "heat_index_f"], inplace=True)
    return df


def add_zone_features(df, target_col):
    df = df.sort_values("datetime_ending_ept").copy()
    df["hour"] = df["datetime_ending_ept"].dt.hour
    df["day_shift"] = np.where(df["hour"].isin(range(1, 10)), 1, 2)

    hist = (
        df[["datetime_ending_ept", target_col]]
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

        df[time_col] = (
            df["datetime_ending_ept"]
            - pd.to_timedelta(df["day_shift"], unit="D")
            - pd.Timedelta(hours=lag)
        )

        df[feat_col] = hist[target_col].reindex(df[time_col]).to_numpy()
        helper_cols.append(time_col)
        feature_cols.append(feat_col)

    hist["MW_roll_24_base"] = hist[target_col].shift(1).rolling(24).mean()
    hist["MW_roll_48_base"] = hist[target_col].shift(1).rolling(48).mean()
    hist["MW_roll_72_base"] = hist[target_col].shift(1).rolling(72).mean()
    hist["MW_roll_168_base"] = hist[target_col].shift(1).rolling(168).mean()
    hist["MW_std_24_base"] = hist[target_col].shift(1).rolling(24).std()
    hist["MW_std_168_base"] = hist[target_col].shift(1).rolling(168).std()

    df["roll_time"] = df["datetime_ending_ept"] - pd.to_timedelta(df["day_shift"], unit="D")

    df["MW_roll_24"] = hist["MW_roll_24_base"].reindex(df["roll_time"]).to_numpy()
    df["MW_roll_48"] = hist["MW_roll_48_base"].reindex(df["roll_time"]).to_numpy()
    df["MW_roll_72"] = hist["MW_roll_72_base"].reindex(df["roll_time"]).to_numpy()
    df["MW_roll_168"] = hist["MW_roll_168_base"].reindex(df["roll_time"]).to_numpy()
    df["MW_std_24"] = hist["MW_std_24_base"].reindex(df["roll_time"]).to_numpy()
    df["MW_std_168"] = hist["MW_std_168_base"].reindex(df["roll_time"]).to_numpy()

    feature_cols += [
        "MW_roll_24", "MW_roll_48", "MW_roll_72", "MW_roll_168",
        "MW_std_24", "MW_std_168"
    ]

    df = df.dropna(subset=feature_cols).copy()
    df.drop(columns=["day_shift", "roll_time"] + helper_cols, inplace=True)
    return df


def add_flags_and_features(df, alert, warning, action):
    df = df.copy()
    df["datetime_ending_ept"] = pd.to_datetime(df["datetime_ending_ept"])

    df = apply_interval_flag(df, alert, "is_alert")
    df = apply_interval_flag(df, warning, "is_warning")
    df = apply_interval_flag(df, action, "is_action")
    df = engineer_common_weather_features(df)

    return df


def decompose_sector_load(df, total_col, hourly_sector_weight, zone_sector_shares):
    df = df.copy()
    df = df.merge(hourly_sector_weight, on="hour", how="left")
    df = df.merge(zone_sector_shares, on="zone", how="left")

    df["wtd_res"] = df["residential_share"] * df["w_res"]
    df["wtd_com"] = df["commercial_share"] * df["w_com"]
    df["wtd_ind"] = df["industrial_share"] * df["w_ind"]

    weight_sum = df[["wtd_res", "wtd_com", "wtd_ind"]].sum(axis=1)

    prefix = "forecast_" if total_col == "forecast_load_mw" else ""
    df[f"res_{prefix}load_mw".replace("__", "_")] = df[total_col] * df["wtd_res"] / weight_sum
    df[f"com_{prefix}load_mw".replace("__", "_")] = df[total_col] * df["wtd_com"] / weight_sum
    df[f"ind_{prefix}load_mw".replace("__", "_")] = df[total_col] * df["wtd_ind"] / weight_sum

    rename_map = {
        "res_load_mw": "res_MW",
        "com_load_mw": "com_MW",
        "ind_load_mw": "ind_MW",
        "res_forecast_load_mw": "res_forecast_load_mw",
        "com_forecast_load_mw": "com_forecast_load_mw",
        "ind_forecast_load_mw": "ind_forecast_load_mw",
    }
    return df.rename(columns=rename_map)


def run_sector_model(sector_df, zone_load_forecast, target_col, forecast_target_col, label_prefix, offset=0):
    results = []
    feat_imp_results = []

    start_date = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize()
    end_date = start_date + pd.Timedelta(days=2) - pd.Timedelta(hours=1)

    zones = sorted(sector_df["zone"].dropna().unique())

    for zone in zones:
        print(f"\n========== Running {label_prefix} zone: {zone} ==========")

        zone_df = sector_df[sector_df["zone"] == zone].sort_values("datetime_ending_ept").copy()
        zone_fcst = zone_load_forecast[zone_load_forecast["zone"] == zone].sort_values("forecast_datetime_ending_ept").copy()

        current_date = start_date

        while current_date <= end_date:
            print(f"Running for zone={zone}, date={current_date.date()}")

            cutoff = (current_date - pd.Timedelta(days=1)).replace(hour=10)

            train_df = zone_df[zone_df["datetime_ending_ept"] <= cutoff].copy()
            test_start = current_date.replace(hour=1)
            test_end = current_date.replace(hour=23) + pd.Timedelta(hours=1)
            test_df = zone_df[
                (zone_df["datetime_ending_ept"] >= test_start) &
                (zone_df["datetime_ending_ept"] <= test_end)
            ].copy()

            if train_df.empty or test_df.empty:
                print("Skipping because train/test is empty")
                current_date += pd.Timedelta(days=1)
                continue

            lower = train_df[target_col].quantile(0.01)
            upper = train_df[target_col].quantile(0.99)
            train_df = train_df[(train_df[target_col] >= lower) & (train_df[target_col] <= upper)].copy()

            drop_cols = [target_col, "datetime_ending_ept", "zone", "date"]
            features = [col for col in train_df.columns if col not in drop_cols]

            valid_hours = 24 * 14
            if len(train_df) <= valid_hours:
                print("Skipping because not enough training history for validation split")
                current_date += pd.Timedelta(days=1)
                continue

            train_part = train_df.iloc[:-valid_hours].copy()
            valid_part = train_df.iloc[-valid_hours:].copy()

            if train_part.empty or valid_part.empty:
                print("Skipping because train_part/valid_part is empty")
                current_date += pd.Timedelta(days=1)
                continue

            X_train = train_part[features]
            y_train = train_part[target_col]
            X_valid = valid_part[features]
            y_valid = valid_part[target_col]
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
                "zone": zone,
                "date": current_date,
                "feature": X_train.columns,
                "importance": model.feature_importances_
            }).sort_values(by="importance", ascending=False)

            feat_imp["rank"] = range(1, len(feat_imp) + 1)
            feat_imp_results.append(feat_imp)

            pred_col = f"{target_col}_pred"
            test_df = test_df.copy()
            test_df[pred_col] = model.predict(X_test)

            forecast_df = zone_fcst[
                (zone_fcst["forecast_datetime_ending_ept"] >= test_start) &
                (zone_fcst["forecast_datetime_ending_ept"] <= test_end)
            ].sort_values("forecast_datetime_ending_ept").copy()

            forecast_df = forecast_df.rename(columns={"forecast_datetime_ending_ept": "datetime_ending_ept"})
            forecast_df["datetime_ending_ept"] = pd.to_datetime(forecast_df["datetime_ending_ept"])

            compare_df = pd.merge(
                test_df,
                forecast_df[["datetime_ending_ept", forecast_target_col]],
                on="datetime_ending_ept",
                how="inner"
            )

            if compare_df.empty:
                print("Skipping because compare_df is empty after forecast merge")
                current_date += pd.Timedelta(days=1)
                continue

            mae = mean_absolute_error(compare_df[forecast_target_col], compare_df[pred_col])
            rmse = np.sqrt(mean_squared_error(compare_df[forecast_target_col], compare_df[pred_col]))

            print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

            compare_df["zone"] = zone
            compare_df["date"] = current_date
            compare_df["MAE"] = mae
            compare_df["RMSE"] = rmse

            results.append(compare_df)
            current_date += pd.Timedelta(days=1)

    final_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    feat_imp_df = pd.concat(feat_imp_results, ignore_index=True) if feat_imp_results else pd.DataFrame()
    return final_df, feat_imp_df


####################################### load forecast ##################################

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
    "BG&E/MIDATL": "BGE", "COMED": "COMED", "DAYTON": "DAYTON", "DEOK": "DEOK",
    "DOMINION": "DOM", "DP&L/MIDATL": "DPL", "DUQUESNE": "DUQ", "EKPC": "EKPC",
    "JCP&L/MIDATL": "JC", "METED/MIDATL": "METED", "PECO/MIDATL": "PE",
    "PENELEC/MIDATL": "PN", "PEPCO/MIDATL": "PEP", "PPL/MIDATL": "PL",
    "PSE&G/MIDATL": "PS", "RECO/MIDATL": "RE", "UGI/MIDATL": "UGI",
    "WESTERN_REGION": "WEST_REGION", "SOUTHERN_REGION": "SOUTH_REGION",
    "MID_ATLANTIC_REGION": "MIDATL_REGION", "RTO_COMBINED": "RTO"
}

load_forecast["zone"] = load_forecast["forecast_area"].map(forecast_to_agg)

load_forecast = (
    load_forecast
    .sort_values(["forecast_datetime_ending_ept", "zone", "evaluated_at_datetime_ept"])
    .drop_duplicates(subset=["forecast_datetime_ending_ept", "zone"], keep="last")
    .reset_index(drop=True)
)

zone_load_forecast = load_forecast[
    load_forecast["zone"].isin([
        "AE", "AEP", "AP", "ATSI", "BGE", "COMED", "DAYTON", "DEOK", "DOM",
        "DPL", "DUQ", "EKPC", "JC", "METED", "PE", "PN", "PEP", "PL", "PS",
        "RE", "UGI"
    ])
].copy()

ugi_mw = (
    zone_load_forecast.loc[zone_load_forecast["zone"] == "UGI", ["forecast_datetime_ending_ept", "forecast_load_mw"]]
    .rename(columns={"forecast_load_mw": "ugi_forecast_load_mw"})
)

zone_load_forecast = zone_load_forecast.merge(ugi_mw, on="forecast_datetime_ending_ept", how="left")

zone_load_forecast.loc[zone_load_forecast["zone"] == "PL", "forecast_load_mw"] = (
    zone_load_forecast.loc[zone_load_forecast["zone"] == "PL", "forecast_load_mw"] +
    zone_load_forecast.loc[zone_load_forecast["zone"] == "PL", "ugi_forecast_load_mw"].fillna(0)
)

zone_load_forecast = (
    zone_load_forecast.loc[zone_load_forecast["zone"] != "UGI"]
    .drop(columns="ugi_forecast_load_mw")
    .reset_index(drop=True)
)

mapping = {
    "AE": "AE", "AEP": "AEP", "AP": "APS", "ATSI": "ATSI", "BGE": "BGE",
    "COMED": "COMED", "DAYTON": "DAYTON", "DEOK": "DUKE", "DOM": "VEPCO",
    "DPL": "DPL", "DUQ": "DQE", "EKPC": "EKPC", "JC": "JCPL",
    "METED": "METED", "PE": "PECO", "PEP": "PEPCO", "PL": "PL",
    "PN": "PENLC", "PS": "PS", "RE": "RECO"
}
zone_load_forecast["zone"] = zone_load_forecast["zone"].map(mapping)

########################################## weather data ############################################

hist_files = [
    "data/weather/weather_hist_2023.csv",
    "data/weather/weather_hist_2024.csv",
    "data/weather/weather_hist_2025.csv",
    "data/weather/weather_hist_2026.csv",
]
forecast_file = f"data/weather/weather_forecast_{run_str}.csv"

hist_weather = pd.concat([pd.read_csv(f) for f in hist_files], ignore_index=True)
hist_weather.drop(columns=["weather_code"], errors="ignore", inplace=True)
hist_weather["time"] = pd.to_datetime(hist_weather["time"])

run_dt = pd.to_datetime(run_str, format="%y%m%d")
hist_weather = hist_weather[hist_weather["time"] <= run_dt].copy()

forecast_weather = pd.read_csv(forecast_file)
forecast_weather.drop(columns=["weather_code"], errors="ignore", inplace=True)
forecast_weather["time"] = pd.to_datetime(forecast_weather["time"])
forecast_weather = forecast_weather[forecast_weather["time"] > run_dt].copy()

dedupe_cols = ["zone", "weather_station", "lat", "lon", "time"]
hist_weather = hist_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
forecast_weather = forecast_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

forecast_weather["hour"] = forecast_weather["time"].dt.hour

weather_vars = [
    "temperature_2m", "apparent_temperature", "dew_point_2m", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "snow_depth", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "wind_direction_10m",
    "wind_gusts_10m", "surface_pressure", "pressure_msl", "et0_fao_evapotranspiration",
    "vapour_pressure_deficit", "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "global_tilted_irradiance", "direct_normal_irradiance", "terrestrial_radiation"
]

hourly_avg = (
    forecast_weather.groupby(["zone", "weather_station", "hour"])[weather_vars]
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

zonal_weather = pd.concat([hist_weather, forecast_weather], ignore_index=True)
zonal_weather = zonal_weather.sort_values(["zone", "weather_station", "time"]).reset_index(drop=True)

for var in weather_vars:
    zonal_weather[f"{var}_weighted"] = zonal_weather[var] * zonal_weather["weather_weight"]

zonal_weather = (
    zonal_weather.groupby(["time", "zone"])[[f"{v}_weighted" for v in weather_vars]]
    .sum()
    .reset_index()
)

zonal_weather = zonal_weather.rename(columns={f"{v}_weighted": v for v in weather_vars})
zonal_weather = zonal_weather.rename(columns={"time": "datetime_ending_ept"})

zone_map = {
    "AE": "AE", "AEP": "AEP", "APS": "AP", "ATSI": "ATSI", "BGE": "BGE",
    "COMED": "COMED", "DAYTON": "DAYTON", "DPL": "DPL", "DQE": "DUQ",
    "EKPC": "EKPC", "JCPL": "JC", "METED": "METED", "PECO": "PE",
    "PENLC": "PN", "PEPCO": "PEP", "PL": "PL", "PS": "PS", "RECO": "RE",
    "UGI": "UGI", "VEPCO": "DOM", "DUKE": "DEOK"
}
zonal_weather["agg_NodeName"] = zonal_weather["zone"].map(zone_map)

#################################### calendar data ###############################################

zonal_weather["datetime_ending_ept"] = pd.to_datetime(zonal_weather["datetime_ending_ept"])
zonal_weather["date"] = zonal_weather["datetime_ending_ept"].dt.normalize()
zonal_weather["year"] = zonal_weather["datetime_ending_ept"].dt.year
zonal_weather["month"] = zonal_weather["datetime_ending_ept"].dt.month
zonal_weather["day"] = zonal_weather["datetime_ending_ept"].dt.day
zonal_weather["hour"] = zonal_weather["datetime_ending_ept"].dt.hour
zonal_weather["day_of_week"] = zonal_weather["datetime_ending_ept"].dt.dayofweek
zonal_weather["is_weekend"] = zonal_weather["day_of_week"].isin([5, 6]).astype(int)

us_holidays = holidays.US(observed=True)
zonal_weather["is_holiday"] = zonal_weather["datetime_ending_ept"].apply(lambda x: int(x.date() in us_holidays))
zonal_weather["WkDayBeforeHol"] = zonal_weather["datetime_ending_ept"].apply(
    lambda x: int((x + pd.Timedelta(days=1)).date() in us_holidays and x.dayofweek < 5)
)
zonal_weather["WkDayAfterHol"] = zonal_weather["datetime_ending_ept"].apply(
    lambda x: int((x - pd.Timedelta(days=1)).date() in us_holidays and x.dayofweek < 5)
)

start_year = zonal_weather["datetime_ending_ept"].dt.year.min()
end_year = zonal_weather["datetime_ending_ept"].dt.year.max()
event_df = build_event_calendar(start_year, end_year)

zonal_weather = zonal_weather.merge(event_df, on="date", how="left")
zonal_weather["is_event"] = zonal_weather["event_name"].notna().astype(int)
zonal_weather.drop(columns=["event_name"], inplace=True)

zonal_wc = zonal_weather[zonal_weather["agg_NodeName"] != "UGI"].copy()

####################################### load data ##########################################################

hourly_load = pd.read_csv(
    f"data/raw_pjm_hrl_load_metered/raw_pjm_hrl_load_metered_{today_str}.csv",
    skiprows=[1],
    usecols=lambda c: c not in ["datetime_beginning_utc", "auto_key", "is_verified", "insert_datetime"]
)

hourly_load["MW"] = pd.to_numeric(hourly_load["MW"], errors="coerce")
hourly_load["datetime_beginning_ept"] = pd.to_datetime(hourly_load["datetime_beginning_ept"]) + pd.Timedelta(hours=1)
hourly_load = hourly_load.rename(columns={"datetime_beginning_ept": "datetime_ending_ept"})

first_cols = ["datetime_ending_ept", "zone"]
hourly_load = hourly_load[first_cols + [c for c in hourly_load.columns if c not in first_cols]]

hourly_zone_load = hourly_load[
    hourly_load["zone"].isin([
        "AE", "AEP", "AP", "ATSI", "BC", "CE", "DAY", "DEOK", "DOM", "DPL",
        "DUQ", "EKPC", "JC", "ME", "OVEC", "PE", "PN", "PEP", "PL", "PS", "RECO"
    ])
].copy()

hourly_zone_load.drop_duplicates(
    subset=["datetime_ending_ept", "zone", "nerc_region", "mkt_region", "load_area"],
    keep="first",
    inplace=True
)

hourly_zone_load = (
    hourly_zone_load.groupby(["datetime_ending_ept", "zone"], as_index=False)
    .agg({"MW": "sum", "nerc_region": "first", "mkt_region": "first"})
)

append_load = pd.read_csv(
    f"data/pjm_All_Instantaneous_Load_rt5/pjm_All_Instantaneous_Load_rt5_{today_str}.csv",
    skiprows=[1]
)

append_load["instantaneous_load"] = pd.to_numeric(append_load["instantaneous_load"], errors="coerce")
append_load["datetime_beginning_ept"] = pd.to_datetime(append_load["datetime_beginning_ept"])
append_load = append_load[append_load["area"] != "UG"].copy()

append_load = (
    append_load
    .assign(
        hour_beginning_ept=lambda x: x["datetime_beginning_ept"].dt.floor("h"),
        datetime_ending_ept=lambda x: x["datetime_beginning_ept"].dt.floor("h") + pd.Timedelta(hours=1)
    )
    .groupby(["area", "datetime_ending_ept"], as_index=False)["instantaneous_load"]
    .mean()
    .rename(columns={"instantaneous_load": "load_mw_hourly_avg"})
)

zone_mapping = {
    "AE": "AE", "AEP": "AEP", "APS": "AP", "ATSI": "ATSI", "BC": "BC", "COMED": "CE",
    "DAYTON": "DAY", "DEOK": "DEOK", "DOM": "DOM", "DPL": "DPL", "DUQ": "DUQ",
    "EKPC": "EKPC", "JC": "JC", "ME": "ME", "PE": "PE", "PEP": "PEP",
    "PL": "PL", "PN": "PN", "PS": "PS", "RECO": "RECO"
}
append_load["zone"] = append_load["area"].map(zone_mapping)
append_load.dropna(subset=["zone"], inplace=True)

max_dt_hourly = hourly_zone_load.groupby("zone")["datetime_ending_ept"].max().min()

append_load_new = append_load.loc[
    append_load["datetime_ending_ept"] > max_dt_hourly,
    ["datetime_ending_ept", "zone", "load_mw_hourly_avg"]
].copy()

append_load_new = append_load_new.rename(columns={"load_mw_hourly_avg": "MW"})
append_load_new["nerc_region"] = append_load_new["zone"]
append_load_new["mkt_region"] = append_load_new["zone"]

hourly_zone_load = pd.concat(
    [hourly_zone_load, append_load_new[["datetime_ending_ept", "zone", "MW", "nerc_region", "mkt_region"]]],
    ignore_index=True
).sort_values("datetime_ending_ept").reset_index(drop=True)

ovec_mw = (
    hourly_zone_load.loc[hourly_zone_load["zone"] == "OVEC", ["datetime_ending_ept", "MW"]]
    .rename(columns={"MW": "OVEC_MW"})
)

hourly_zone_load = hourly_zone_load.merge(ovec_mw, on="datetime_ending_ept", how="left")
hourly_zone_load.loc[hourly_zone_load["zone"] == "AEP", "MW"] = (
    hourly_zone_load.loc[hourly_zone_load["zone"] == "AEP", "MW"] +
    hourly_zone_load.loc[hourly_zone_load["zone"] == "AEP", "OVEC_MW"].fillna(0)
)

hourly_zone_load = (
    hourly_zone_load.loc[hourly_zone_load["zone"] != "OVEC"]
    .drop(columns="OVEC_MW")
    .reset_index(drop=True)
)

zone_mapping = {
    "AE": "AE", "AEP": "AEP", "AP": "APS", "ATSI": "ATSI", "BC": "BGE",
    "CE": "COMED", "DAY": "DAYTON", "DEOK": "DUKE", "DOM": "VEPCO",
    "DPL": "DPL", "DUQ": "DQE", "EKPC": "EKPC", "JC": "JCPL", "ME": "METED",
    "PE": "PECO", "PEP": "PEPCO", "PL": "PL", "PN": "PENLC", "PS": "PS", "RECO": "RECO"
}
hourly_zone_load["zone"] = hourly_zone_load["zone"].map(zone_mapping)

mw_region = hourly_zone_load[["datetime_ending_ept", "zone", "MW"]].copy()
zonal_wcl = zonal_wc.merge(mw_region, on=["datetime_ending_ept", "zone"], how="left")

######################################## sector decomposition #############################

hourly_sector_weight = pd.read_csv("data/hrl_sct_wt.csv")

zone_sector_shares = pd.DataFrame([
    {"zone": "AE", "residential_share": 0.45, "commercial_share": 0.45, "industrial_share": 0.09},
    {"zone": "AEP", "residential_share": 0.35, "commercial_share": 0.29, "industrial_share": 0.37},
    {"zone": "APS", "residential_share": 0.39, "commercial_share": 0.21, "industrial_share": 0.40},
    {"zone": "ATSI", "residential_share": 0.34, "commercial_share": 0.26, "industrial_share": 0.40},
    {"zone": "BGE", "residential_share": 0.45, "commercial_share": 0.51, "industrial_share": 0.04},
    {"zone": "COMED", "residential_share": 0.33, "commercial_share": 0.36, "industrial_share": 0.32},
    {"zone": "DAYTON", "residential_share": 0.38, "commercial_share": 0.33, "industrial_share": 0.29},
    {"zone": "DUKE", "residential_share": 0.38, "commercial_share": 0.38, "industrial_share": 0.24},
    {"zone": "VEPCO", "residential_share": 0.34, "commercial_share": 0.56, "industrial_share": 0.10},
    {"zone": "DPL", "residential_share": 0.48, "commercial_share": 0.37, "industrial_share": 0.15},
    {"zone": "DQE", "residential_share": 0.33, "commercial_share": 0.46, "industrial_share": 0.21},
    {"zone": "EKPC", "residential_share": 0.55, "commercial_share": 0.14, "industrial_share": 0.31},
    {"zone": "JCPL", "residential_share": 0.49, "commercial_share": 0.42, "industrial_share": 0.09},
    {"zone": "METED", "residential_share": 0.42, "commercial_share": 0.15, "industrial_share": 0.43},
    {"zone": "PECO", "residential_share": 0.39, "commercial_share": 0.22, "industrial_share": 0.39},
    {"zone": "PEPCO", "residential_share": 0.39, "commercial_share": 0.59, "industrial_share": 0.02},
    {"zone": "PL", "residential_share": 0.40, "commercial_share": 0.38, "industrial_share": 0.23},
    {"zone": "PENLC", "residential_share": 0.37, "commercial_share": 0.19, "industrial_share": 0.45},
    {"zone": "PS", "residential_share": 0.35, "commercial_share": 0.56, "industrial_share": 0.09},
    {"zone": "RECO", "residential_share": 0.49, "commercial_share": 0.50, "industrial_share": 0.01},
])

zone_load_forecast["forecast_datetime_ending_ept"] = pd.to_datetime(zone_load_forecast["forecast_datetime_ending_ept"])
zone_load_forecast["hour"] = zone_load_forecast["forecast_datetime_ending_ept"].dt.hour

zone_load_forecast = decompose_sector_load(
    zone_load_forecast,
    total_col="forecast_load_mw",
    hourly_sector_weight=hourly_sector_weight,
    zone_sector_shares=zone_sector_shares
)

zonal_wcl = decompose_sector_load(
    zonal_wcl,
    total_col="MW",
    hourly_sector_weight=hourly_sector_weight,
    zone_sector_shares=zone_sector_shares
)

feature_cols = [
    "datetime_ending_ept", "zone",
    "temperature_2m", "apparent_temperature", "dew_point_2m",
    "relative_humidity_2m", "precipitation", "rain", "snowfall", "snow_depth",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "surface_pressure", "pressure_msl", "et0_fao_evapotranspiration", "vapour_pressure_deficit",
    "shortwave_radiation", "direct_radiation", "diffuse_radiation",
    "global_tilted_irradiance", "direct_normal_irradiance", "terrestrial_radiation",
    "date", "year", "month", "day", "hour", "day_of_week", "is_weekend",
    "is_holiday", "WkDayBeforeHol", "WkDayAfterHol", "is_event"
]

res_zone_load = zonal_wcl[["datetime_ending_ept", "zone", "res_MW"] + feature_cols[2:]].copy()
com_zone_load = zonal_wcl[["datetime_ending_ept", "zone", "com_MW"] + feature_cols[2:]].copy()
ind_zone_load = zonal_wcl[["datetime_ending_ept", "zone", "ind_MW"] + feature_cols[2:]].copy()

############################################### event intervals ###################################################

alert = pd.read_csv("data/emergencymessages/alerts.csv")
alert["start"] = pd.to_datetime(alert["start"].astype(str).str[:-6])
alert["end"] = pd.to_datetime(alert["end"].astype(str).str[:-6])

warning = pd.read_csv("data/emergencymessages/warnings.csv")
warning["start"] = pd.to_datetime(warning["start"].astype(str).str[:-6])
warning["end"] = pd.to_datetime(warning["end"].astype(str).str[:-6])

action = pd.read_csv("data/emergencymessages/actions.csv")
action["start"] = pd.to_datetime(action["start"].astype(str).str[:-6])
action["end"] = pd.to_datetime(action["end"].astype(str).str[:-6])

############################################### residential ###################################################

res_zone_load = add_flags_and_features(res_zone_load, alert, warning, action)
res_zone_load = pd.concat(
    [add_zone_features(g.copy(), "res_MW") for _, g in res_zone_load.groupby("zone")],
    ignore_index=True
)

res_final, res_final_feat_importance = run_sector_model(
    sector_df=res_zone_load,
    zone_load_forecast=zone_load_forecast,
    target_col="res_MW",
    forecast_target_col="res_forecast_load_mw",
    label_prefix="res",
    offset=offset
)

############################################### commercial ###################################################

com_zone_load = add_flags_and_features(com_zone_load, alert, warning, action)
com_zone_load = pd.concat(
    [add_zone_features(g.copy(), "com_MW") for _, g in com_zone_load.groupby("zone")],
    ignore_index=True
)

com_final, com_final_feat_importance = run_sector_model(
    sector_df=com_zone_load,
    zone_load_forecast=zone_load_forecast,
    target_col="com_MW",
    forecast_target_col="com_forecast_load_mw",
    label_prefix="com",
    offset=offset
)

############################################### industrial ###################################################

ind_zone_load = add_flags_and_features(ind_zone_load, alert, warning, action)
ind_zone_load = pd.concat(
    [add_zone_features(g.copy(), "ind_MW") for _, g in ind_zone_load.groupby("zone")],
    ignore_index=True
)

ind_final, ind_final_feat_importance = run_sector_model(
    sector_df=ind_zone_load,
    zone_load_forecast=zone_load_forecast,
    target_col="ind_MW",
    forecast_target_col="ind_forecast_load_mw",
    label_prefix="ind",
    offset=offset
)

############################################## sector union ###################################################

zone_load_forecast = zone_load_forecast.rename(columns={"forecast_datetime_ending_ept": "datetime_ending_ept"})

pred_df = (
    res_final[["datetime_ending_ept", "zone", "res_MW_pred"]]
    .merge(com_final[["datetime_ending_ept", "zone", "com_MW_pred"]], on=["datetime_ending_ept", "zone"], how="outer")
    .merge(ind_final[["datetime_ending_ept", "zone", "ind_MW_pred"]], on=["datetime_ending_ept", "zone"], how="outer")
)

pred_df["total_MW_pred"] = (
    pred_df["res_MW_pred"].fillna(0) +
    pred_df["com_MW_pred"].fillna(0) +
    pred_df["ind_MW_pred"].fillna(0)
)

pred_df = pred_df.merge(
    zonal_wcl[["datetime_ending_ept", "zone", "MW"]],
    on=["datetime_ending_ept", "zone"],
    how="left"
)

pred_df = pred_df.merge(
    zone_load_forecast[["datetime_ending_ept", "zone", "forecast_load_mw"]],
    on=["datetime_ending_ept", "zone"],
    how="left"
)

pred_df["date"] = pred_df["datetime_ending_ept"].dt.date
pred_df["hour"] = pred_df["datetime_ending_ept"].dt.hour

start_str = today.strftime("%m%d")
end_str = (today + timedelta(days=1)).strftime("%m%d")

pred_df[
    ["datetime_ending_ept", "zone", "date", "hour", "total_MW_pred", "forecast_load_mw"]
].to_csv(
    f"data/prediction/zone_sector_prediction_{start_str}_to_{end_str}_at_{run_str}_try.csv",
    index=False
)