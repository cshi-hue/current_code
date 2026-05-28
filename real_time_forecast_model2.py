import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm

from datetime import datetime, timedelta

import holidays

import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

############################ load forecast data ##########################################

offset = 0
today_str = (datetime.today() - timedelta(days=offset)).strftime("%m%d") 

load_forecast = pd.read_csv(f'data/load_frcstd_7_day/load_frcstd_7_day_{today_str}.csv', skiprows=[1])

load_forecast['forecast_load_mw'] = pd.to_numeric(load_forecast['forecast_load_mw'], errors='coerce')

load_forecast = load_forecast[['evaluated_at_datetime_ept', 'forecast_datetime_ending_ept', 'forecast_area', 'forecast_load_mw']].copy()

load_forecast["forecast_datetime_ending_ept"] = pd.to_datetime(load_forecast["forecast_datetime_ending_ept"])

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
    
    # Regions
    "WESTERN_REGION": "WEST_REGION",
    "SOUTHERN_REGION": "SOUTH_REGION",
    "MID_ATLANTIC_REGION": "MIDATL_REGION",

    # Whole system
    "RTO_COMBINED": "RTO"
}

load_forecast["agg_NodeName"] = load_forecast["forecast_area"].map(forecast_to_agg)

load_forecast = (
    load_forecast
    .sort_values(["forecast_datetime_ending_ept", "agg_NodeName", "evaluated_at_datetime_ept"])
    .drop_duplicates(subset=["forecast_datetime_ending_ept", "agg_NodeName"], keep="last")
    .reset_index(drop=True)
)

rto_load_forecast = load_forecast[load_forecast['agg_NodeName'] == 'RTO'].copy()

rto_load_forecast['forecast_datetime_ending_ept'] = pd.to_datetime(rto_load_forecast['forecast_datetime_ending_ept'])
rto_load_forecast['hour'] = rto_load_forecast['forecast_datetime_ending_ept'].dt.hour
rto_load_forecast.rename(columns={"agg_NodeName": "zone"}, inplace=True)


############################### weather data ##########################################

run_str = (datetime.today() - timedelta(days=offset)).strftime("%y%m%d")  

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

# keep only hist data up to run_str
run_dt = pd.to_datetime(run_str, format="%y%m%d")
hist_weather = hist_weather[hist_weather["datetime_ending_ept"] <= run_dt].copy()

# --- load forecast ---
forecast_weather = pd.read_csv(forecast_file)
forecast_weather.drop(columns=["weather_code"], errors="ignore", inplace=True)
forecast_weather.rename(columns={"time": "datetime_ending_ept"}, inplace=True)
forecast_weather["datetime_ending_ept"] = pd.to_datetime(forecast_weather["datetime_ending_ept"])

# keep only forecast data from forecast_run_str
forecast_weather = forecast_weather[
    forecast_weather["datetime_ending_ept"] > run_dt
].copy()

# dedupe
dedupe_cols = ["zone", "weather_station", "lat", "lon", "datetime_ending_ept"]
hist_weather = hist_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
forecast_weather = forecast_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

# --- fill forecast nulls with hour average ---
forecast_weather["hour"] = forecast_weather["datetime_ending_ept"].dt.hour

weather_vars = [
    'temperature_2m', 
    'apparent_temperature', 
    'dew_point_2m',
    'relative_humidity_2m', 
    'precipitation',
    'rain', 
    'snowfall', 
    'snow_depth', 
    'cloud_cover', 
    'cloud_cover_low',
    'cloud_cover_mid', 
    'cloud_cover_high', 
    'wind_speed_10m',
    'wind_direction_10m', 
    'wind_gusts_10m', 
    'surface_pressure', 
    'pressure_msl',
    'et0_fao_evapotranspiration',
    'vapour_pressure_deficit',
    'shortwave_radiation', 
    'direct_radiation', 
    'diffuse_radiation',
    'global_tilted_irradiance', 
    'direct_normal_irradiance',
    'terrestrial_radiation'
]

# hour average within each zone/station/hour
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

# drop helper columns
drop_cols = (
    ["hour"]
    + [f"{c}_hour_avg" for c in weather_vars]
)
forecast_weather.drop(columns=drop_cols, inplace=True, errors="ignore")

# --- final zonal_weather ---
zonal_weather = pd.concat([hist_weather, forecast_weather], ignore_index=True)
zonal_weather = zonal_weather.sort_values(
    ["zone", "weather_station", "datetime_ending_ept"]
).reset_index(drop=True)

for var in weather_vars:
    zonal_weather[f"{var}_weighted"] = (
        zonal_weather[var] * zonal_weather["weather_weight"]
    )

zonal_weather = (
    zonal_weather.groupby(["datetime_ending_ept", "zone"])[
        [f"{v}_weighted" for v in weather_vars]
    ]
    .sum()
    .reset_index()
)

zonal_weather = zonal_weather.rename(columns={
    "temperature_2m_weighted": "temperature_2m",
    "apparent_temperature_weighted": "apparent_temperature",
    "dew_point_2m_weighted": "dew_point_2m",
    "relative_humidity_2m_weighted": "relative_humidity_2m",
    "precipitation_weighted": "precipitation",
    "rain_weighted": "rain",
    "snowfall_weighted": "snowfall",
    "snow_depth_weighted": "snow_depth",
    "cloud_cover_weighted": "cloud_cover",
    "cloud_cover_low_weighted": "cloud_cover_low",
    "cloud_cover_mid_weighted": "cloud_cover_mid",
    "cloud_cover_high_weighted": "cloud_cover_high",
    "wind_speed_10m_weighted": "wind_speed_10m",
    "wind_direction_10m_weighted": "wind_direction_10m",
    "wind_gusts_10m_weighted": "wind_gusts_10m",
    "surface_pressure_weighted": "surface_pressure",
    "pressure_msl_weighted": "pressure_msl",
    "et0_fao_evapotranspiration_weighted": "et0_fao_evapotranspiration",
    "vapour_pressure_deficit_weighted": "vapour_pressure_deficit",
    'shortwave_radiation_weighted': 'shortwave_radiation',
    'direct_radiation_weighted': 'direct_radiation',
    'diffuse_radiation_weighted': 'diffuse_radiation',
    'global_tilted_irradiance_weighted': 'global_tilted_irradiance',
    'direct_normal_irradiance_weighted': 'direct_normal_irradiance',
    'terrestrial_radiation_weighted': 'terrestrial_radiation'
})


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

zonal_weather['datetime_ending_ept'] = pd.to_datetime(zonal_weather['datetime_ending_ept'])

zonal_weather['date'] = zonal_weather['datetime_ending_ept'].dt.date
zonal_weather['year'] = zonal_weather['datetime_ending_ept'].dt.year
zonal_weather['month'] = zonal_weather['datetime_ending_ept'].dt.month
zonal_weather['day'] = zonal_weather['datetime_ending_ept'].dt.day
zonal_weather['hour'] = zonal_weather['datetime_ending_ept'].dt.hour
zonal_weather['day_of_week'] = zonal_weather['datetime_ending_ept'].dt.dayofweek

zonal_weather['is_weekend'] = zonal_weather['day_of_week'].isin([5,6]).astype(int)

us_holidays = holidays.US(observed=True)

zonal_weather['is_holiday'] = zonal_weather['datetime_ending_ept'].apply(
    lambda x: int(x.date() in us_holidays)
)

zonal_weather['WkDayBeforeHol'] = zonal_weather['datetime_ending_ept'].apply(
    lambda x: int((x + pd.Timedelta(days=1)) in us_holidays and pd.Timestamp(x).dayofweek < 5)
)

zonal_weather['WkDayAfterHol'] = zonal_weather['datetime_ending_ept'].apply(
    lambda x: int((x - pd.Timedelta(days=1)) in us_holidays and pd.Timestamp(x).dayofweek < 5)
)

from dateutil.easter import easter

def build_event_calendar(start_year, end_year):
    rows = []

    # Federal holidays
    us = holidays.US(years=range(start_year, end_year + 1), observed=True)
    for d, name in us.items():
        rows.append((pd.Timestamp(d), name))

    for y in range(start_year, end_year + 1):
        # Easter
        e = pd.Timestamp(easter(y))
        rows += [
            (e - pd.Timedelta(days=2), "Good Friday"),
            (e, "Easter"),
        ]

        # Fixed-date events 
        rows += [
            (pd.Timestamp(f"{y}-12-24"), "Christmas Eve"),
            (pd.Timestamp(f"{y}-12-31"), "New Year's Eve"),
            (pd.Timestamp(f"{y}-10-31"), "Halloween")
        ]

        # Thanksgiving-based 
        tg = [d for d, n in holidays.US(years=[y]).items() if "Thanksgiving" in n][0]

        rows += [
            (pd.Timestamp(tg) + pd.Timedelta(days=1), "Black Friday"),
            (pd.Timestamp(tg) + pd.Timedelta(days=4), "Cyber Monday"),
        ]

    event_df = pd.DataFrame(rows, columns=["date", "event_name"])

    # combine duplicates
    event_df = (
        event_df.groupby("date")["event_name"]
        .apply(lambda x: ", ".join(sorted(set(x))))
        .reset_index()
        .sort_values("date")
    )

    return event_df

# build calendar
start_year = zonal_weather["datetime_ending_ept"].dt.year.min()
end_year   = zonal_weather["datetime_ending_ept"].dt.year.max()

event_df = build_event_calendar(start_year, end_year)

# prepare join key
zonal_weather["date"] = pd.to_datetime(zonal_weather["datetime_ending_ept"]).dt.normalize()

# merge
zonal_weather = zonal_weather.merge(event_df, on="date", how="left")

# flag
zonal_weather["is_event"] = zonal_weather["event_name"].notna().astype(int)

zonal_weather.drop(columns=["event_name"], inplace=True)

zonal_wc = zonal_weather.copy()

zonal_wc = zonal_wc[zonal_wc['agg_NodeName'] != 'UGI'].copy()

########################### load data ##########################################

# read data
hourly_load = pd.read_csv(f'data/raw_pjm_hrl_load_metered/raw_pjm_hrl_load_metered_{today_str}.csv', skiprows=[1], usecols=lambda c: c not in ["datetime_beginning_utc", "auto_key", "is_verified", "insert_datetime"])

hourly_load["MW"] = pd.to_numeric(hourly_load["MW"], errors='coerce')

hourly_load["datetime_beginning_ept"] = pd.to_datetime(hourly_load["datetime_beginning_ept"])
hourly_load["datetime_beginning_ept"] += pd.Timedelta(hours=1)

hourly_load = hourly_load.rename(columns={
    "datetime_beginning_ept": "datetime_ending_ept"
})

# desired first columns
first_cols = ["datetime_ending_ept", "zone"]

# reorder dataframe
hourly_load = hourly_load[first_cols + [c for c in hourly_load.columns if c not in first_cols]]

hourly_rto_load = hourly_load[hourly_load['zone'].isin(['RTO'])].copy()

hourly_rto_load = (
    hourly_rto_load
    .groupby(['datetime_ending_ept', 'zone'], as_index=False)
    .agg({
        'MW': 'first',
        'nerc_region': 'first',
        'mkt_region': 'first',
    })
)

append_load = pd.read_csv(f"data/pjm_All_Instantaneous_Load_rt5/pjm_All_Instantaneous_Load_rt5_{today_str}.csv", skiprows=[1])

append_load["instantaneous_load"] = pd.to_numeric(append_load["instantaneous_load"], errors='coerce')

append_load["datetime_beginning_ept"] = pd.to_datetime(append_load["datetime_beginning_ept"])

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

weather_cols = [
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
    'shortwave_radiation', 
    'direct_radiation', 
    'diffuse_radiation',
    'global_tilted_irradiance', 
    'direct_normal_irradiance',
    'terrestrial_radiation'
]

other_cols = [c for c in zonal_wc.columns if c not in weather_cols + ["agg_NodeName"]]

# aggregation dictionary
agg_dict = {}

# mean (keeps original names)
for col in weather_cols:
    agg_dict[col] = ["median", "std"]

# other columns
agg_dict.update({col: "first" for col in other_cols if col != "datetime_ending_ept"})

# groupby
rto_weather = (
    zonal_wc
    .groupby("datetime_ending_ept", as_index=False)
    .agg(agg_dict)
)

# flatten MultiIndex columns
rto_weather.columns = [
    col if isinstance(col, str) else
    f"{col[0]}_std" if col[1] == "std" else col[0]
    for col in rto_weather.columns
]

rto_weather.drop(columns=["zone"], inplace=True)

# merge
rto_load = pd.merge(
    hourly_rto_load,
    rto_weather,
    on="datetime_ending_ept",
    how='right'   
)

rto_load.drop(columns=["mkt_region", "nerc_region"], inplace=True)

# sort just in case
rto_load = rto_load.sort_values('datetime_ending_ept')

rto_load["datetime_ending_ept"] = pd.to_datetime(rto_load["datetime_ending_ept"])

# hour of target timestamp
rto_load["hour"] = rto_load["datetime_ending_ept"].dt.hour

################################### RTO sector prediction ##################################

hourly_sector_weight = pd.read_csv('data/hrl_sct_wt.csv')

rto_sector_shares = pd.DataFrame([{"zone": "RTO",  "residential_share": 0.37, "commercial_share": 0.37, "industrial_share": 0.25}])

df = rto_load_forecast.copy()

df = df.merge(hourly_sector_weight, on="hour", how="left")

df = df.merge(rto_sector_shares, on="zone", how="left")

# unnormalized
df["wtd_res"] = df["residential_share"] * df["w_res"]
df["wtd_com"]  = df["commercial_share"]  * df["w_com"]
df["wtd_ind"]  = df["industrial_share"]  * df["w_ind"]

# normalize
weight_sum = df[["wtd_res", "wtd_com", "wtd_ind"]].sum(axis=1)

df["res_forecast_load_mw"] = df["forecast_load_mw"] * df["wtd_res"] / weight_sum
df["com_forecast_load_mw"]  = df["forecast_load_mw"] * df["wtd_com"]  / weight_sum
df["ind_forecast_load_mw"]  = df["forecast_load_mw"] * df["wtd_ind"]  / weight_sum

rto_load_forecast = df.copy()

df = rto_load.copy()

df = df.merge(hourly_sector_weight, on="hour", how="left")

df = df.merge(rto_sector_shares, on="zone", how="left")

# unnormalized
df["wtd_res"] = df["residential_share"] * df["w_res"]
df["wtd_com"]  = df["commercial_share"]  * df["w_com"]
df["wtd_ind"]  = df["industrial_share"]  * df["w_ind"]

# normalize
weight_sum = df[["wtd_res", "wtd_com", "wtd_ind"]].sum(axis=1)

df["res_MW"] = df["MW"] * df["wtd_res"] / weight_sum
df["com_MW"]  = df["MW"] * df["wtd_com"]  / weight_sum
df["ind_MW"]  = df["MW"] * df["wtd_ind"]  / weight_sum

res_rto_load = df[[
    'datetime_ending_ept', 'zone', 'res_MW',
    'temperature_2m', 'apparent_temperature', 'dew_point_2m', 
    'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'snow_depth', 
    'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'surface_pressure', 'pressure_msl', 'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'global_tilted_irradiance', 'direct_normal_irradiance', 'terrestrial_radiation',
    'temperature_2m_std', 'apparent_temperature_std', 'dew_point_2m_std', 
    'relative_humidity_2m_std', 'precipitation_std', 'rain_std', 'snowfall_std', 'snow_depth_std', 
    'cloud_cover_std', 'cloud_cover_low_std', 'cloud_cover_mid_std', 'cloud_cover_high_std',
    'wind_speed_10m_std', 'wind_direction_10m_std', 'wind_gusts_10m_std',
    'surface_pressure_std', 'pressure_msl_std', 'et0_fao_evapotranspiration_std', 'vapour_pressure_deficit_std',
    'shortwave_radiation_std', 'direct_radiation_std', 'diffuse_radiation_std', 'global_tilted_irradiance_std', 'direct_normal_irradiance_std', 'terrestrial_radiation_std',
    'date', 'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_holiday', 'WkDayBeforeHol', 'WkDayAfterHol', 'is_event'
]].copy()

com_rto_load = df[[
    'datetime_ending_ept', 'zone', 'com_MW',
    'temperature_2m', 'apparent_temperature', 'dew_point_2m', 
    'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'snow_depth', 
    'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'surface_pressure', 'pressure_msl', 'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'global_tilted_irradiance', 'direct_normal_irradiance', 'terrestrial_radiation',
    'temperature_2m_std', 'apparent_temperature_std', 'dew_point_2m_std', 
    'relative_humidity_2m_std', 'precipitation_std', 'rain_std', 'snowfall_std', 'snow_depth_std', 
    'cloud_cover_std', 'cloud_cover_low_std', 'cloud_cover_mid_std', 'cloud_cover_high_std',
    'wind_speed_10m_std', 'wind_direction_10m_std', 'wind_gusts_10m_std',
    'surface_pressure_std', 'pressure_msl_std', 'et0_fao_evapotranspiration_std', 'vapour_pressure_deficit_std',
    'shortwave_radiation_std', 'direct_radiation_std', 'diffuse_radiation_std', 'global_tilted_irradiance_std', 'direct_normal_irradiance_std', 'terrestrial_radiation_std',
    'date', 'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_holiday', 'WkDayBeforeHol', 'WkDayAfterHol', 'is_event'
]].copy()

ind_rto_load = df[[
    'datetime_ending_ept', 'zone', 'ind_MW',
    'temperature_2m', 'apparent_temperature', 'dew_point_2m', 
    'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'snow_depth', 
    'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'surface_pressure', 'pressure_msl', 'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'global_tilted_irradiance', 'direct_normal_irradiance', 'terrestrial_radiation',
    'temperature_2m_std', 'apparent_temperature_std', 'dew_point_2m_std', 
    'relative_humidity_2m_std', 'precipitation_std', 'rain_std', 'snowfall_std', 'snow_depth_std', 
    'cloud_cover_std', 'cloud_cover_low_std', 'cloud_cover_mid_std', 'cloud_cover_high_std',
    'wind_speed_10m_std', 'wind_direction_10m_std', 'wind_gusts_10m_std',
    'surface_pressure_std', 'pressure_msl_std', 'et0_fao_evapotranspiration_std', 'vapour_pressure_deficit_std',
    'shortwave_radiation_std', 'direct_radiation_std', 'diffuse_radiation_std', 'global_tilted_irradiance_std', 'direct_normal_irradiance_std', 'terrestrial_radiation_std',
    'date', 'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_holiday', 'WkDayBeforeHol', 'WkDayAfterHol', 'is_event'
]].copy()

################################## residential ########################################
# make sure datetime types match
res_rto_load["datetime_ending_ept"] = pd.to_datetime(res_rto_load["datetime_ending_ept"])

alert = pd.read_csv("data/emergencymessages/alerts.csv")
alert["start"] = pd.to_datetime(alert["start"].astype(str).str[:-6])
alert["end"]   = pd.to_datetime(alert["end"].astype(str).str[:-6])

# now your existing code works
res_rto_load["is_alert"] = 0

for _, row in alert.iterrows():
    mask = (
        (res_rto_load["datetime_ending_ept"] >= row["start"]) &
        (res_rto_load["datetime_ending_ept"] <= row["end"])
    )
    res_rto_load.loc[mask, "is_alert"] = 1

warning = pd.read_csv("data/emergencymessages/warnings.csv")
warning["start"] = pd.to_datetime(warning["start"].astype(str).str[:-6])
warning["end"]   = pd.to_datetime(warning["end"].astype(str).str[:-6])

res_rto_load["is_warning"] = 0

for _, row in warning.iterrows():
    mask = (
        (res_rto_load["datetime_ending_ept"] >= row["start"]) &
        (res_rto_load["datetime_ending_ept"] <= row["end"])
    )
    res_rto_load.loc[mask, "is_warning"] = 1

action = pd.read_csv("data/emergencymessages/actions.csv")
action["start"] = pd.to_datetime(action["start"].astype(str).str[:-6])
action["end"]   = pd.to_datetime(action["end"].astype(str).str[:-6])

res_rto_load["is_action"] = 0

for _, row in action.iterrows():
    mask = (
        (res_rto_load["datetime_ending_ept"] >= row["start"]) &
        (res_rto_load["datetime_ending_ept"] <= row["end"])
    )
    res_rto_load.loc[mask, "is_action"] = 1

# temp
daily_temp = (
    res_rto_load.groupby("date", as_index=False)["temperature_2m"]
    .agg(temp_min="min", temp_max="max")
)

res_rto_load = res_rto_load.merge(daily_temp, on="date", how="left")

res_rto_load["temp_f"] = res_rto_load["temperature_2m"] * 9/5 + 32

base_temp = 65

res_rto_load["HDD"] = np.maximum(base_temp - res_rto_load["temp_f"], 0)
res_rto_load["CDD"] = np.maximum(res_rto_load["temp_f"] - base_temp, 0)

res_rto_load["HDD_wind"] = res_rto_load["HDD"] * res_rto_load["wind_speed_10m"]
res_rto_load["CDD_cloud"] = res_rto_load["CDD"] * (1 - res_rto_load["cloud_cover"] / 100)

res_rto_load["wind_dir_10m_sin"] = np.sin(np.deg2rad(res_rto_load["wind_direction_10m"]))
res_rto_load["wind_dir_10m_cos"] = np.cos(np.deg2rad(res_rto_load["wind_direction_10m"]))

res_rto_load["wind_chill"] = (
    35.74
    + 0.6215 * res_rto_load["temp_f"]
    - 35.75 * (res_rto_load["wind_speed_10m"] ** 0.16)
    + 0.4275 * res_rto_load["temp_f"] * (res_rto_load["wind_speed_10m"] ** 0.16)
)

RH = res_rto_load["relative_humidity_2m"]
T = res_rto_load["temp_f"]

res_rto_load["heat_index_f"] = (
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

res_rto_load["feels_like_temp"] = np.where(
    (res_rto_load["temp_f"] <= 50) & (res_rto_load["wind_speed_10m"] > 3),
    res_rto_load["wind_chill"],
    np.where(
        (res_rto_load["temp_f"] >= 80) & (RH >= 40),
        res_rto_load["heat_index_f"],
        res_rto_load["temp_f"]
    )
)

# hour-to-hour absolute temperature change
res_rto_load["temp_diff_1h_abs"] = res_rto_load["temperature_2m"].diff().abs()

# cumulative total variation over past 6 hours
res_rto_load["temp_total_variation_6h"] = (
    res_rto_load["temp_diff_1h_abs"].rolling(6).sum()
)

# cumulative total variation over past 12 hours
res_rto_load["temp_total_variation_12h"] = (
    res_rto_load["temp_diff_1h_abs"].rolling(12).sum()
)

# cumulative total variation over past 24 hours
res_rto_load["temp_total_variation_24h"] = (
    res_rto_load["temp_diff_1h_abs"].rolling(24).sum()
)

res_rto_load["temp_fcst_1h"] = res_rto_load["temperature_2m"].shift(-1)

horizon = 6

res_rto_load["temp_lead_mean_6h"] = sum(
    res_rto_load["temperature_2m"].shift(-i)
    for i in range(1, horizon + 1)
) / horizon


res_rto_load.drop(columns=["wind_chill", "temp_diff_1h_abs", "heat_index_f"], inplace=True)

res_rto_load = res_rto_load.sort_values("datetime_ending_ept").copy()
res_rto_load["datetime_ending_ept"] = pd.to_datetime(res_rto_load["datetime_ending_ept"])

# hour of target timestamp
res_rto_load["hour"] = res_rto_load["datetime_ending_ept"].dt.hour

# if target hour < 10, use 1 day back
# if target hour >= 10, use 2 days back
res_rto_load["day_shift"] = np.where(res_rto_load["hour"].isin(range(1,10)), 1, 2)

# lookup index from historical MW
hist = (
    res_rto_load[["datetime_ending_ept", "res_MW"]]
    .drop_duplicates("datetime_ending_ept")
    .sort_values("datetime_ending_ept")
    .set_index("datetime_ending_ept")
)

# choose lag hours to create
lag_hours = [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168]

# build lag lookup timestamps + features
helper_cols = []
feature_cols = []

for lag in lag_hours:
    time_col = f"lag{lag}_time"
    feat_col = f"MW_lag_{lag}"

    res_rto_load[time_col] = (
        res_rto_load["datetime_ending_ept"]
        - pd.to_timedelta(res_rto_load["day_shift"], unit="D")
        - pd.Timedelta(hours=lag)
    )

    res_rto_load[feat_col] = hist["res_MW"].reindex(res_rto_load[time_col]).to_numpy()

    helper_cols.append(time_col)
    feature_cols.append(feat_col)

# rolling stats based only on historical series
hist["MW_roll_24_base"] = hist["res_MW"].shift(1).rolling(24).mean()
hist["MW_roll_48_base"] = hist["res_MW"].shift(1).rolling(48).mean()
hist["MW_roll_72_base"] = hist["res_MW"].shift(1).rolling(72).mean()
hist["MW_roll_168_base"] = hist["res_MW"].shift(1).rolling(168).mean()

hist["MW_std_24_base"] = hist["res_MW"].shift(1).rolling(24).std()
hist["MW_std_168_base"] = hist["res_MW"].shift(1).rolling(168).std()

# rolling lookup time
res_rto_load["roll_time"] = (
    res_rto_load["datetime_ending_ept"]
    - pd.to_timedelta(res_rto_load["day_shift"], unit="D")
)

res_rto_load["MW_roll_24"] = hist["MW_roll_24_base"].reindex(res_rto_load["roll_time"]).to_numpy()
res_rto_load["MW_roll_48"] = hist["MW_roll_48_base"].reindex(res_rto_load["roll_time"]).to_numpy()
res_rto_load["MW_roll_72"] = hist["MW_roll_72_base"].reindex(res_rto_load["roll_time"]).to_numpy()
res_rto_load["MW_roll_168"] = hist["MW_roll_168_base"].reindex(res_rto_load["roll_time"]).to_numpy()

res_rto_load["MW_std_24"] = hist["MW_std_24_base"].reindex(res_rto_load["roll_time"]).to_numpy()
res_rto_load["MW_std_168"] = hist["MW_std_168_base"].reindex(res_rto_load["roll_time"]).to_numpy()

feature_cols += [
    "MW_roll_24", "MW_roll_48", "MW_roll_72", "MW_roll_168",
    "MW_std_24", "MW_std_168"
]

# drop rows without enough history
res_rto_load = res_rto_load.dropna(subset=feature_cols).copy()

# optional: remove helper columns
res_rto_load.drop(columns=["day_shift", "roll_time"] + helper_cols, inplace=True)

rto_load_forecast.rename(columns={"forecast_datetime_ending_ept": "datetime_ending_ept"}, inplace=True)

rto_load_forecast["datetime_ending_ept"] = pd.to_datetime(
        rto_load_forecast["datetime_ending_ept"]
    )

res_rto_load = pd.merge(
    res_rto_load,
    rto_load_forecast[["datetime_ending_ept", "res_forecast_load_mw"]],
    on="datetime_ending_ept",
    how="left"
)

results = []
feat_imp_results = []

start_date = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() # pd.Timestamp("2026-03-25")
end_date   = start_date + pd.Timedelta(days=2) - pd.Timedelta(hours=1) # pd.Timestamp("2026-03-26")

current_date = start_date

while current_date <= end_date:

    print(f"Running for {current_date.date()}")

    # Define cutoff (day before 10:00)
    cutoff = (current_date - pd.Timedelta(days=1)).replace(hour=10)

    # Train set
    train_df = res_rto_load[
        res_rto_load["datetime_ending_ept"] <= cutoff
    ].copy()

    # Test set (full target day)
    test_start = current_date.replace(hour=1)
    test_end   = current_date.replace(hour=23) + pd.Timedelta(hours=1)

    test_df = res_rto_load[
        (res_rto_load["datetime_ending_ept"] >= test_start) &
        (res_rto_load["datetime_ending_ept"] <= test_end)
    ].copy()

    # Target / features
    target = "res_MW"
    drop_cols = ["res_MW", "datetime_ending_ept", "zone", "date", "res_forecast_load_mw"]
    features = [col for col in train_df.columns if col not in drop_cols]

    # Time-based validation split
    valid_hours = 24 * 14
    train_part = train_df.iloc[:-valid_hours].copy()
    valid_part = train_df.iloc[-valid_hours:].copy()

    X_train = train_part[features]
    y_train = train_part[target]

    X_valid = valid_part[features]
    y_valid = valid_part[target]

    X_test = test_df[features]

    # Build weights based on PJM forecast miss > 5%
    train_dev_pct = (
        (train_part["res_MW"] - train_part["res_forecast_load_mw"]).abs()
        / train_part["res_forecast_load_mw"].abs().clip(lower=1e-6)
    )
    valid_dev_pct = (
        (valid_part["res_MW"] - valid_part["res_forecast_load_mw"]).abs()
        / valid_part["res_forecast_load_mw"].abs().clip(lower=1e-6)
    )

    train_extreme = train_dev_pct > 0.02
    valid_extreme = valid_dev_pct > 0.02

    # heavier penalty on extreme periods
    train_weights = np.where(train_extreme, 2, 1)
    valid_weights = np.where(valid_extreme, 2, 1)

    # Custom eval metric: RMSE only on extreme periods
    valid_forecast = valid_part["res_forecast_load_mw"].to_numpy()

    def extreme_rmse_eval(y_true, y_pred):
        extreme_mask = (
            np.abs(y_true - valid_forecast) /
            np.clip(np.abs(valid_forecast), 1e-6, None)
        ) > 0.02

        if extreme_mask.sum() == 0:
            return ("extreme_rmse", 0.0, False)

        rmse = np.sqrt(np.mean((y_true[extreme_mask] - y_pred[extreme_mask]) ** 2))
        return ("extreme_rmse", rmse, False)

    # Train model
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

    model.fit(
        X_train,
        y_train,
        sample_weight=train_weights,
        eval_set=[(X_valid, y_valid)],
        eval_sample_weight=[valid_weights],
        eval_metric=lambda yt, yp: [
            extreme_rmse_eval(yt, yp)
        ],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

    # Feature importance
    feat_imp = pd.DataFrame({
        "date": current_date,
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feat_imp["rank"] = range(1, len(feat_imp) + 1)
    feat_imp_results.append(feat_imp)

    # Predict
    test_df["res_MW_pred"] = model.predict(X_test)

    # Metrics on test day
    mae = mean_absolute_error(test_df["res_forecast_load_mw"], test_df["res_MW_pred"])
    rmse = np.sqrt(mean_squared_error(test_df["res_forecast_load_mw"], test_df["res_MW_pred"]))

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    test_df["date"] = current_date
    test_df["MAE"] = mae
    test_df["RMSE"] = rmse

    results.append(test_df)

    current_date += pd.Timedelta(days=1)

# combine all days
res_final = pd.concat(results).reset_index(drop=True)

############################# commercial ############################################

# make sure datetime types match
com_rto_load["datetime_ending_ept"] = pd.to_datetime(com_rto_load["datetime_ending_ept"])

# alert 
com_rto_load["is_alert"] = 0

for _, row in alert.iterrows():
    mask = (
        (com_rto_load["datetime_ending_ept"] >= row["start"]) &
        (com_rto_load["datetime_ending_ept"] <= row["end"])
    )
    com_rto_load.loc[mask, "is_alert"] = 1

com_rto_load["is_warning"] = 0

for _, row in warning.iterrows():
    mask = (
        (com_rto_load["datetime_ending_ept"] >= row["start"]) &
        (com_rto_load["datetime_ending_ept"] <= row["end"])
    )
    com_rto_load.loc[mask, "is_warning"] = 1

com_rto_load["is_action"] = 0

for _, row in action.iterrows():
    mask = (
        (com_rto_load["datetime_ending_ept"] >= row["start"]) &
        (com_rto_load["datetime_ending_ept"] <= row["end"])
    )
    com_rto_load.loc[mask, "is_action"] = 1


# temp
daily_temp = (
    com_rto_load.groupby("date", as_index=False)["temperature_2m"]
    .agg(temp_min="min", temp_max="max")
)

com_rto_load = com_rto_load.merge(daily_temp, on="date", how="left")

com_rto_load["temp_f"] = com_rto_load["temperature_2m"] * 9/5 + 32

base_temp = 65

com_rto_load["HDD"] = np.maximum(base_temp - com_rto_load["temp_f"], 0)
com_rto_load["CDD"] = np.maximum(com_rto_load["temp_f"] - base_temp, 0)

com_rto_load["wind_chill"] = (
    35.74
    + 0.6215 * com_rto_load["temp_f"]
    - 35.75 * (com_rto_load["wind_speed_10m"] ** 0.16)
    + 0.4275 * com_rto_load["temp_f"] * (com_rto_load["wind_speed_10m"] ** 0.16)
)

RH = com_rto_load["relative_humidity_2m"]
T = com_rto_load["temp_f"]

com_rto_load["heat_index_f"] = (
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

com_rto_load["feels_like_temp"] = np.where(
    (com_rto_load["temp_f"] <= 50) & (com_rto_load["wind_speed_10m"] > 3),
    com_rto_load["wind_chill"],
    np.where(
        (com_rto_load["temp_f"] >= 80) & (RH >= 40),
        com_rto_load["heat_index_f"],
        com_rto_load["temp_f"]
    )
)

com_rto_load["wind_dir_10m_sin"] = np.sin(np.deg2rad(com_rto_load["wind_direction_10m"]))
com_rto_load["wind_dir_10m_cos"] = np.cos(np.deg2rad(com_rto_load["wind_direction_10m"]))

com_rto_load["HDD_wind"] = com_rto_load["HDD"] * com_rto_load["wind_speed_10m"]
com_rto_load["CDD_cloud"] = com_rto_load["CDD"] * (1 - com_rto_load["cloud_cover"] / 100)

# hour-to-hour absolute temperature change
com_rto_load["temp_diff_1h_abs"] = com_rto_load["temperature_2m"].diff().abs()

# cumulative total variation over past 6 hours
com_rto_load["temp_total_variation_6h"] = (
    com_rto_load["temp_diff_1h_abs"].rolling(6).sum()
)

# cumulative total variation over past 12 hours
com_rto_load["temp_total_variation_12h"] = (
    com_rto_load["temp_diff_1h_abs"].rolling(12).sum()
)

# cumulative total variation over past 24 hours
com_rto_load["temp_total_variation_24h"] = (
    com_rto_load["temp_diff_1h_abs"].rolling(24).sum()
)

com_rto_load.drop(columns=["wind_chill", "temp_diff_1h_abs", "heat_index_f"], inplace=True)

# if target hour < 10, use 1 day back
# if target hour >= 10, use 2 days back
com_rto_load["day_shift"] = np.where(com_rto_load["hour"].isin(range(1, 10)), 1, 2)

# lookup index
hist = (
    com_rto_load[["datetime_ending_ept", "com_MW"]]
    .drop_duplicates("datetime_ending_ept")
    .sort_values("datetime_ending_ept")
    .set_index("datetime_ending_ept")
)

# choose lag hours to create
lag_hours = [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168]

# build lag lookup timestamps + features
helper_cols = []
feature_cols = []

for lag in lag_hours:
    time_col = f"lag{lag}_time"
    feat_col = f"MW_lag_{lag}"

    com_rto_load[time_col] = (
        com_rto_load["datetime_ending_ept"]
        - pd.to_timedelta(com_rto_load["day_shift"], unit="D")
        - pd.Timedelta(hours=lag)
    )

    com_rto_load[feat_col] = hist["com_MW"].reindex(com_rto_load[time_col]).to_numpy()

    helper_cols.append(time_col)
    feature_cols.append(feat_col)

# rolling stats based only on historical series
hist["MW_roll_24_base"] = hist["com_MW"].shift(1).rolling(24).mean()
hist["MW_roll_48_base"] = hist["com_MW"].shift(1).rolling(48).mean()
hist["MW_roll_72_base"] = hist["com_MW"].shift(1).rolling(72).mean()
hist["MW_roll_168_base"] = hist["com_MW"].shift(1).rolling(168).mean()

hist["MW_std_24_base"] = hist["com_MW"].shift(1).rolling(24).std()
hist["MW_std_168_base"] = hist["com_MW"].shift(1).rolling(168).std()

# rolling lookup time
com_rto_load["roll_time"] = (
    com_rto_load["datetime_ending_ept"]
    - pd.to_timedelta(com_rto_load["day_shift"], unit="D")
)

com_rto_load["MW_roll_24"] = hist["MW_roll_24_base"].reindex(com_rto_load["roll_time"]).to_numpy()
com_rto_load["MW_roll_48"] = hist["MW_roll_48_base"].reindex(com_rto_load["roll_time"]).to_numpy()
com_rto_load["MW_roll_72"] = hist["MW_roll_72_base"].reindex(com_rto_load["roll_time"]).to_numpy()
com_rto_load["MW_roll_168"] = hist["MW_roll_168_base"].reindex(com_rto_load["roll_time"]).to_numpy()

com_rto_load["MW_std_24"] = hist["MW_std_24_base"].reindex(com_rto_load["roll_time"]).to_numpy()
com_rto_load["MW_std_168"] = hist["MW_std_168_base"].reindex(com_rto_load["roll_time"]).to_numpy()

feature_cols += [
    "MW_roll_24", "MW_roll_48", "MW_roll_72", "MW_roll_168",
    "MW_std_24", "MW_std_168"
]

# drop rows without enough history
com_rto_load = com_rto_load.dropna(subset=feature_cols).copy()

# optional: remove helper columns
com_rto_load.drop(columns=["day_shift", "roll_time"] + helper_cols, inplace=True)


results = []
feat_imp_results = []

start_date = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() # pd.Timestamp("2026-03-25")
end_date   = start_date + pd.Timedelta(days=2) - pd.Timedelta(hours=1) # pd.Timestamp("2026-03-26")

current_date = start_date

while current_date <= end_date:

    print(f"Running for {current_date.date()}")

    # Define cutoff (day before 10:00)
    cutoff = (current_date - pd.Timedelta(days=1)).replace(hour=10)

    # Train set
    train_df = com_rto_load[
        com_rto_load["datetime_ending_ept"] <= cutoff
    ].copy()

    #  Test set (full target day)
    test_start = current_date.replace(hour=1)
    test_end   = current_date.replace(hour=23) + pd.Timedelta(hours=1)

    test_df = com_rto_load[
        (com_rto_load["datetime_ending_ept"] >= test_start) &
        (com_rto_load["datetime_ending_ept"] <= test_end)
    ].copy()

    # Features
    target = "com_MW"

    drop_cols = ["com_MW", "datetime_ending_ept", "zone", "date"]
    features = [col for col in train_df.columns if col not in drop_cols]

    # Time-based validation split from training data
    # use the most recent 14 days of hourly data as validation
    valid_hours = 24 * 14
    
    train_part = train_df.iloc[:-valid_hours].copy()
    valid_part = train_df.iloc[-valid_hours:].copy()

    X_train = train_part[features]
    y_train = train_part[target]

    X_valid = valid_part[features]
    y_valid = valid_part[target]

    X_test = test_df[features]

    # Train model with recommended params + early stopping
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
            __import__("lightgbm").early_stopping(100),
            __import__("lightgbm").log_evaluation(100)
        ]
    )

    # feature importance for this day 
    feat_imp = pd.DataFrame({
        "date": current_date,
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feat_imp["rank"] = range(1, len(feat_imp) + 1)
    feat_imp_results.append(feat_imp)


    # Predict
    test_df["com_MW_pred"] = model.predict(X_test)

    # Merge with PJM forecast
    forecast_df = rto_load_forecast[
        (rto_load_forecast["datetime_ending_ept"] >= test_start) &
        (rto_load_forecast["datetime_ending_ept"] <= test_end)
    ].copy()

    compare_df = pd.merge(
        test_df,
        forecast_df[["datetime_ending_ept", "com_forecast_load_mw"]],
        on="datetime_ending_ept",
        how="left"
    )

    # Metrics
    mae = mean_absolute_error(compare_df["com_forecast_load_mw"], compare_df["com_MW_pred"])
    rmse = np.sqrt(mean_squared_error(compare_df["com_forecast_load_mw"], compare_df["com_MW_pred"]))

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    compare_df["date"] = current_date
    compare_df["MAE"] = mae
    compare_df["RMSE"] = rmse

    results.append(compare_df)

    # move to next day
    current_date += pd.Timedelta(days=1)

# combine all days
com_final = pd.concat(results).reset_index(drop=True)

################################## industrial ######################################

# make sure datetime types match
ind_rto_load["datetime_ending_ept"] = pd.to_datetime(ind_rto_load["datetime_ending_ept"])

ind_rto_load["is_alert"] = 0

for _, row in alert.iterrows():
    mask = (
        (ind_rto_load["datetime_ending_ept"] >= row["start"]) &
        (ind_rto_load["datetime_ending_ept"] <= row["end"])
    )
    ind_rto_load.loc[mask, "is_alert"] = 1


ind_rto_load["is_warning"] = 0

for _, row in warning.iterrows():
    mask = (
        (ind_rto_load["datetime_ending_ept"] >= row["start"]) &
        (ind_rto_load["datetime_ending_ept"] <= row["end"])
    )
    ind_rto_load.loc[mask, "is_warning"] = 1

ind_rto_load["is_action"] = 0

for _, row in action.iterrows():
    mask = (
        (ind_rto_load["datetime_ending_ept"] >= row["start"]) &
        (ind_rto_load["datetime_ending_ept"] <= row["end"])
    )
    ind_rto_load.loc[mask, "is_action"] = 1

# temp
daily_temp = (
    ind_rto_load.groupby("date", as_index=False)["temperature_2m"]
    .agg(temp_min="min", temp_max="max")
)

ind_rto_load = ind_rto_load.merge(daily_temp, on="date", how="left")

ind_rto_load["temp_f"] = ind_rto_load["temperature_2m"] * 9/5 + 32

base_temp = 65

ind_rto_load["HDD"] = np.maximum(base_temp - ind_rto_load["temp_f"], 0)
ind_rto_load["CDD"] = np.maximum(ind_rto_load["temp_f"] - base_temp, 0)

ind_rto_load["wind_chill"] = (
    35.74
    + 0.6215 * ind_rto_load["temp_f"]
    - 35.75 * (ind_rto_load["wind_speed_10m"] ** 0.16)
    + 0.4275 * ind_rto_load["temp_f"] * (ind_rto_load["wind_speed_10m"] ** 0.16)
)

RH = ind_rto_load["relative_humidity_2m"]
T = ind_rto_load["temp_f"]

ind_rto_load["heat_index_f"] = (
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

ind_rto_load["feels_like_temp"] = np.where(
    (ind_rto_load["temp_f"] <= 50) & (ind_rto_load["wind_speed_10m"] > 3),
    ind_rto_load["wind_chill"],
    np.where(
        (ind_rto_load["temp_f"] >= 80) & (RH >= 40),
        ind_rto_load["heat_index_f"],
        ind_rto_load["temp_f"]
    )
)

ind_rto_load["wind_dir_10m_sin"] = np.sin(np.deg2rad(ind_rto_load["wind_direction_10m"]))
ind_rto_load["wind_dir_10m_cos"] = np.cos(np.deg2rad(ind_rto_load["wind_direction_10m"]))

ind_rto_load["HDD_wind"] = ind_rto_load["HDD"] * ind_rto_load["wind_speed_10m"]
ind_rto_load["CDD_cloud"] = ind_rto_load["CDD"] * (1 - ind_rto_load["cloud_cover"] / 100)


# hour-to-hour absolute temperature change
ind_rto_load["temp_diff_1h_abs"] = ind_rto_load["temperature_2m"].diff().abs()

# cumulative total variation over past 6 hours
ind_rto_load["temp_total_variation_6h"] = (
    ind_rto_load["temp_diff_1h_abs"].rolling(6).sum()
)

# cumulative total variation over past 12 hours
ind_rto_load["temp_total_variation_12h"] = (
    ind_rto_load["temp_diff_1h_abs"].rolling(12).sum()
)

# cumulative total variation over past 24 hours
ind_rto_load["temp_total_variation_24h"] = (
    ind_rto_load["temp_diff_1h_abs"].rolling(24).sum()
)

ind_rto_load.drop(columns=["wind_chill", "temp_diff_1h_abs", "heat_index_f"], inplace=True)

# if target hour < 10, use 1 day back
# if target hour >= 10, use 2 days back
ind_rto_load["day_shift"] = np.where(ind_rto_load["hour"].isin(range(1, 10)), 1, 2)

# lookup index from historical MW
hist = (
    ind_rto_load[["datetime_ending_ept", "ind_MW"]]
    .drop_duplicates("datetime_ending_ept")
    .sort_values("datetime_ending_ept")
    .set_index("datetime_ending_ept")
)

# choose lag hours to create
lag_hours = [1, 2, 3, 6, 12, 24, 36, 48, 72, 96, 120, 144, 168]

# build lag lookup timestamps + features
helper_cols = []
feature_cols = []

for lag in lag_hours:
    time_col = f"lag{lag}_time"
    feat_col = f"MW_lag_{lag}"

    ind_rto_load[time_col] = (
        ind_rto_load["datetime_ending_ept"]
        - pd.to_timedelta(ind_rto_load["day_shift"], unit="D")
        - pd.Timedelta(hours=lag)
    )

    ind_rto_load[feat_col] = hist["ind_MW"].reindex(ind_rto_load[time_col]).to_numpy()

    helper_cols.append(time_col)
    feature_cols.append(feat_col)

# rolling stats based only on historical series
hist["MW_roll_24_base"] = hist["ind_MW"].shift(1).rolling(24).mean()
hist["MW_roll_48_base"] = hist["ind_MW"].shift(1).rolling(48).mean()
hist["MW_roll_72_base"] = hist["ind_MW"].shift(1).rolling(72).mean()
hist["MW_roll_168_base"] = hist["ind_MW"].shift(1).rolling(168).mean()

hist["MW_std_24_base"] = hist["ind_MW"].shift(1).rolling(24).std()
hist["MW_std_168_base"] = hist["ind_MW"].shift(1).rolling(168).std()

# rolling lookup time
ind_rto_load["roll_time"] = (
    ind_rto_load["datetime_ending_ept"]
    - pd.to_timedelta(ind_rto_load["day_shift"], unit="D")
)

ind_rto_load["MW_roll_24"] = hist["MW_roll_24_base"].reindex(ind_rto_load["roll_time"]).to_numpy()
ind_rto_load["MW_roll_48"] = hist["MW_roll_48_base"].reindex(ind_rto_load["roll_time"]).to_numpy()
ind_rto_load["MW_roll_72"] = hist["MW_roll_72_base"].reindex(ind_rto_load["roll_time"]).to_numpy()
ind_rto_load["MW_roll_168"] = hist["MW_roll_168_base"].reindex(ind_rto_load["roll_time"]).to_numpy()

ind_rto_load["MW_std_24"] = hist["MW_std_24_base"].reindex(ind_rto_load["roll_time"]).to_numpy()
ind_rto_load["MW_std_168"] = hist["MW_std_168_base"].reindex(ind_rto_load["roll_time"]).to_numpy()

feature_cols += [
    "MW_roll_24", "MW_roll_48", "MW_roll_72", "MW_roll_168",
    "MW_std_24", "MW_std_168"
]

# drop rows without enough history
ind_rto_load = ind_rto_load.dropna(subset=feature_cols).copy()

# optional: remove helper columns
ind_rto_load.drop(columns=["day_shift", "roll_time"] + helper_cols, inplace=True)


results = []
feat_imp_results = []

start_date = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() # pd.Timestamp("2026-03-25")
end_date   = start_date + pd.Timedelta(days=2) - pd.Timedelta(hours=1) # pd.Timestamp("2026-03-26")


current_date = start_date

while current_date <= end_date:

    print(f"Running for {current_date.date()}")

    # 1. Define cutoff (day before 10:00)
    cutoff = (current_date - pd.Timedelta(days=1)).replace(hour=10)

    # 2. Train set
    train_df = ind_rto_load[
        ind_rto_load["datetime_ending_ept"] <= cutoff
    ].copy()

    # 3. Test set (full target day)
    test_start = current_date.replace(hour=1)
    test_end   = current_date.replace(hour=23) + pd.Timedelta(hours=1)

    test_df = ind_rto_load[
        (ind_rto_load["datetime_ending_ept"] >= test_start) &
        (ind_rto_load["datetime_ending_ept"] <= test_end)
    ].copy()

    # 4. Features
    target = "ind_MW"

    drop_cols = ["ind_MW", "datetime_ending_ept", "zone", "date"]
    features = [col for col in train_df.columns if col not in drop_cols]

    # 5. Time-based validation split from training data
    # use the most recent 14 days of hourly data as validation
    valid_hours = 24 * 14
    
    train_part = train_df.iloc[:-valid_hours].copy()
    valid_part = train_df.iloc[-valid_hours:].copy()

    X_train = train_part[features]
    y_train = train_part[target]

    X_valid = valid_part[features]
    y_valid = valid_part[target]

    X_test = test_df[features]

    # 6. Train model with recommended params + early stopping
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
            __import__("lightgbm").early_stopping(100),
            __import__("lightgbm").log_evaluation(100)
        ]
    )

    # ---- feature importance for this day ----
    feat_imp = pd.DataFrame({
        "date": current_date,
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    feat_imp["rank"] = range(1, len(feat_imp) + 1)
    feat_imp_results.append(feat_imp)

    # 6. Predict
    test_df["ind_MW_pred"] = model.predict(X_test)

    # Merge with PJM forecast
    forecast_df = rto_load_forecast[
        (rto_load_forecast["datetime_ending_ept"] >= test_start) &
        (rto_load_forecast["datetime_ending_ept"] <= test_end)
    ].copy()

    compare_df = pd.merge(
        test_df,
        forecast_df[["datetime_ending_ept", "ind_forecast_load_mw"]],
        on="datetime_ending_ept",
        how="inner"
    )

    # Metrics
    mae = mean_absolute_error(compare_df["ind_forecast_load_mw"], compare_df["ind_MW_pred"])
    rmse = np.sqrt(mean_squared_error(compare_df["ind_forecast_load_mw"], compare_df["ind_MW_pred"]))

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    compare_df["date"] = current_date
    compare_df["MAE"] = mae
    compare_df["RMSE"] = rmse

    results.append(compare_df)

    # move to next day
    current_date += pd.Timedelta(days=1)

# combine all days
ind_final = pd.concat(results).reset_index(drop=True)

################################################# sector union ######################################

rto_load_forecast.rename(columns={"forecast_datetime_ending_ept": "datetime_ending_ept"}, inplace=True)

# merge predictions together
pred_df = (
    res_final[["datetime_ending_ept" , "res_MW_pred"]]
    .merge(
        com_final[["datetime_ending_ept", "com_MW_pred"]],
        on="datetime_ending_ept",
        how="outer"
    )
    .merge(
        ind_final[["datetime_ending_ept", "ind_MW_pred"]],
        on="datetime_ending_ept",
        how="outer"
    )
)

pred_df["total_MW_pred"] = (
    pred_df["res_MW_pred"]
    + pred_df["com_MW_pred"]
    + pred_df["ind_MW_pred"]
)

pred_df = (
    pred_df
    .merge(
        rto_load[["datetime_ending_ept", "MW"]],
        on="datetime_ending_ept",
        how="left"
    )
)

pred_df = (
    pred_df
    .merge(
        rto_load_forecast[["datetime_ending_ept", "forecast_load_mw"]],
        on="datetime_ending_ept",
        how="left"
    )
)

# extract date
pred_df["date"] = pred_df["datetime_ending_ept"].dt.date
pred_df['hour'] = pred_df["datetime_ending_ept"].dt.hour

daily_metrics = pred_df.groupby("date").apply(
    lambda df: pd.Series({
        "mae": mean_absolute_error(df["forecast_load_mw"], df["total_MW_pred"]),
        "rmse": np.sqrt(mean_squared_error(df["forecast_load_mw"], df["total_MW_pred"]))
    })
).reset_index()

today = datetime.today() - timedelta(days=offset)

start_str = today.strftime("%m%d")                 # 0325
end_str = (today + timedelta(days=1)).strftime("%m%d")  # 0326
run_str = today.strftime("%y%m%d")                 # 260325

pred_df[
    ['datetime_ending_ept', 'date', 'hour', 'total_MW_pred', 'forecast_load_mw']
].to_csv(
    f'data/prediction/RTO_sector_prediction_{start_str}_to_{end_str}_at_{run_str}.csv',
    index=False
)
