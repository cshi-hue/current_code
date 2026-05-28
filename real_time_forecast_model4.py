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

####################################### load forecast ##################################
offset = 0
today_str = (datetime.today() - timedelta(days=offset)).strftime("%m%d") 

load_forecast = pd.read_csv(f'data/load_frcstd_7_day/load_frcstd_7_day_{today_str}.csv', skiprows=[1])

load_forecast['forecast_load_mw'] = pd.to_numeric(load_forecast['forecast_load_mw'], errors='coerce')

load_forecast =  load_forecast[['evaluated_at_datetime_ept', 'forecast_datetime_ending_ept', 'forecast_area', 'forecast_load_mw']].copy()

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

load_forecast["zone"] = load_forecast["forecast_area"].map(forecast_to_agg)

load_forecast = (
    load_forecast
    .sort_values(["forecast_datetime_ending_ept", "zone", "evaluated_at_datetime_ept"])
    .drop_duplicates(subset=["forecast_datetime_ending_ept", "zone"], keep="last")
    .reset_index(drop=True)
)

zone_load_forecast = load_forecast[load_forecast['zone'].isin(['AE', 'AEP', 'AP', 'ATSI', 'BGE', 'COMED', 'DAYTON', 'DEOK', 'DOM', 'DPL', 'DUQ', 'EKPC', 'JC', 'METED', 'PE', 'PN', 'PEP', 'PL', 'PS', 'RE'])].copy()

# # get OVEC MW by timestamp
# ugi_mw = (
#     zone_load_forecast.loc[zone_load_forecast["zone"] == "UGI", ["forecast_datetime_ending_ept", "forecast_load_mw"]]
#     .rename(columns={"forecast_load_mw": "ugi_forecast_load_mw"})
# )

# # attach OVEC MW to matching rows
# zone_load_forecast = zone_load_forecast.merge(ugi_mw, on="forecast_datetime_ending_ept", how="left")

# # add OVEC MW only to AEP rows
# zone_load_forecast.loc[zone_load_forecast["zone"] == "PL", "forecast_load_mw"] = (
#     zone_load_forecast.loc[zone_load_forecast["zone"] == "PL", "forecast_load_mw"] +
#     zone_load_forecast.loc[zone_load_forecast["zone"] == "PL", "ugi_forecast_load_mw"].fillna(0)
# )

# # remove OVEC rows and helper column
# zone_load_forecast = (
#     zone_load_forecast.loc[zone_load_forecast["zone"] != "UGI"]
#     .drop(columns="ugi_forecast_load_mw")
#     .reset_index(drop=True)
# )

mapping = {
    'AE': 'AE',
    'AEP': 'AEP',
    'AP': 'APS',
    'ATSI': 'ATSI',
    'BGE': 'BGE',
    'COMED': 'COMED',
    'DAYTON': 'DAYTON',
    'DEOK': 'DUKE',
    'DOM': 'VEPCO',
    'DPL': 'DPL',
    'DUQ': 'DQE',
    'EKPC': 'EKPC',
    'JC': 'JCPL',
    'METED': 'METED',
    'PE': 'PECO',
    'PEP': 'PEPCO',
    'PL': 'PL',
    'PN': 'PENLC',
    'PS': 'PS',
    'RE': 'RECO'
}

zone_load_forecast['zone'] = zone_load_forecast['zone'].map(mapping)

########################################## weather data ############################################

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
hist_weather["time"] = pd.to_datetime(hist_weather["time"])

# keep only hist data up to run_str
run_dt = pd.to_datetime(run_str, format="%y%m%d")
hist_weather = hist_weather[hist_weather["time"] <= run_dt].copy()

# --- load forecast ---
forecast_weather = pd.read_csv(forecast_file)
forecast_weather.drop(columns=["weather_code"], errors="ignore", inplace=True)
forecast_weather["time"] = pd.to_datetime(forecast_weather["time"])

# keep only forecast data from forecast_run_str
forecast_weather = forecast_weather[
    forecast_weather["time"] > run_dt
].copy()

# dedupe
dedupe_cols = ["zone", "weather_station", "lat", "lon", "time"]
hist_weather = hist_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)
forecast_weather = forecast_weather.drop_duplicates(subset=dedupe_cols, keep="first").reset_index(drop=True)

# --- fill forecast nulls with hour average ---
forecast_weather["hour"] = forecast_weather["time"].dt.hour

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
    ["zone", "weather_station", "time"]
).reset_index(drop=True)

for var in weather_vars:
    zonal_weather[f"{var}_weighted"] = (
        zonal_weather[var] * zonal_weather["weather_weight"]
    )

zonal_weather = (
    zonal_weather.groupby(["time", "zone"])[
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

zonal_weather.rename(columns={"time": "datetime_ending_ept"}, inplace=True)

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

#################################### calendar data ###############################################

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

####################################### load data ##########################################################
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

hourly_zone_load = hourly_load[hourly_load['zone'].isin(['AE', 'AEP', 'AP', 'ATSI', 'BC', 'CE', 'DAY', 'DEOK', 'DOM', 'DPL', 'DUQ', 'EKPC', 'JC', 'ME', 'OVEC', 'PE', 'PN', 'PEP', 'PL', 'PS', 'RECO'])].copy()

hourly_zone_load.drop_duplicates(subset=['datetime_ending_ept', 'zone', 'nerc_region', 'mkt_region', 'load_area'], keep='first', inplace=True)

hourly_zone_load = (
    hourly_zone_load
    .groupby(['datetime_ending_ept', 'zone'], as_index=False)
    .agg({
        'MW': 'sum',
        'nerc_region': 'first',
        'mkt_region': 'first',
    })
)

append_load = pd.read_csv(f"data/pjm_All_Instantaneous_Load_rt5/pjm_All_Instantaneous_Load_rt5_{today_str}.csv", skiprows=[1])

append_load["instantaneous_load"] = pd.to_numeric(append_load["instantaneous_load"], errors='coerce')

append_load["datetime_beginning_ept"] = pd.to_datetime(append_load["datetime_beginning_ept"])

append_load = append_load[append_load['area'] != 'UG'].copy()

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
    'AE': 'AE',
    'AEP': 'AEP',
    'APS': 'AP',
    'ATSI': 'ATSI',
    'BC': 'BC',
    'COMED': 'CE',
    'DAYTON': 'DAY',
    'DEOK': 'DEOK',
    'DOM': 'DOM',
    'DPL': 'DPL',
    'DUQ': 'DUQ',
    'EKPC': 'EKPC',
    'JC': 'JC',
    'ME': 'ME',
    'PE': 'PE',
    'PEP': 'PEP',
    'PL': 'PL',
    'PN': 'PN',
    'PS': 'PS',
    'RECO': 'RECO'
}

append_load["zone"] = append_load["area"].map(zone_mapping)

append_load.dropna(subset=["zone"], inplace=True)

# find the latest timestamp already in hourly_zone_load
max_dt_per_zone = (
    hourly_zone_load
    .groupby("zone")["datetime_ending_ept"]
    .max()
)

max_dt_hourly = max_dt_per_zone.min()

# keep only rows from append_load that are after that timestamp
append_load_new = append_load.loc[
    append_load["datetime_ending_ept"] > max_dt_hourly,
    ["datetime_ending_ept", "zone", "load_mw_hourly_avg"]
].copy()

# rename to match hourly_zone_load schema
append_load_new = append_load_new.rename(columns={"load_mw_hourly_avg": "MW"})
append_load_new["nerc_region"] = append_load_new["zone"]
append_load_new["mkt_region"] = append_load_new["zone"]

# append to hourly_zone_load
hourly_zone_load = pd.concat(
    [hourly_zone_load, append_load_new[["datetime_ending_ept", "zone", "MW", "nerc_region", "mkt_region"]]],
    ignore_index=True
).sort_values("datetime_ending_ept").reset_index(drop=True)

# get OVEC MW by timestamp
ovec_mw = (
    hourly_zone_load.loc[hourly_zone_load["zone"] == "OVEC", ["datetime_ending_ept", "MW"]]
    .rename(columns={"MW": "OVEC_MW"})
)

# attach OVEC MW to matching rows
hourly_zone_load = hourly_zone_load.merge(ovec_mw, on="datetime_ending_ept", how="left")

# add OVEC MW only to AEP rows
hourly_zone_load.loc[hourly_zone_load["zone"] == "AEP", "MW"] = (
    hourly_zone_load.loc[hourly_zone_load["zone"] == "AEP", "MW"] +
    hourly_zone_load.loc[hourly_zone_load["zone"] == "AEP", "OVEC_MW"].fillna(0)
)

# remove OVEC rows and helper column
hourly_zone_load = (
    hourly_zone_load.loc[hourly_zone_load["zone"] != "OVEC"]
    .drop(columns="OVEC_MW")
    .reset_index(drop=True)
)

zone_mapping = {
    "AE": "AE",
    "AEP": "AEP",
    "AP": "APS",
    "ATSI": "ATSI",
    "BC": "BGE",
    "CE": "COMED",
    "DAY": "DAYTON",
    "DEOK": "DUKE", 
    "DOM": "VEPCO",
    "DPL": "DPL",
    "DUQ": "DQE",
    "EKPC": "EKPC",
    "JC": "JCPL",
    "ME": "METED",
    "PE": "PECO",
    "PEP": "PEPCO",
    "PL": "PL",
    "PN": "PENLC",
    "PS": "PS",
    "RECO": "RECO"
}

hourly_zone_load["zone"] = hourly_zone_load["zone"].map(zone_mapping)

# keep only the columns you want to bring into zonal_wc
mw_region = hourly_zone_load[["datetime_ending_ept", "zone", "MW"]].copy()

# merge into zonal_wc
zonal_wcl = zonal_wc.merge(
    mw_region,
    on=["datetime_ending_ept", "zone"],
    how="left"
)

######################################## zone cutoff #####################################
# zonal_wcl = zonal_wcl[zonal_wcl['zone'].isin(['VEPCO', 'BGE', 'AEP', 'PENLC'])].copy()

######################################## sector decomposition #############################

hourly_sector_weight = pd.read_csv('data/hrl_sct_wt.csv')

zone_sector_shares = pd.DataFrame([
    {"zone": "AE",     "residential_share": 0.45, "commercial_share": 0.45, "industrial_share": 0.09},
    {"zone": "AEP",    "residential_share": 0.35, "commercial_share": 0.29, "industrial_share": 0.37},
    {"zone": "APS",    "residential_share": 0.39, "commercial_share": 0.21, "industrial_share": 0.40},
    {"zone": "ATSI",   "residential_share": 0.34, "commercial_share": 0.26, "industrial_share": 0.40},
    {"zone": "BGE",    "residential_share": 0.45, "commercial_share": 0.51, "industrial_share": 0.04},
    {"zone": "COMED",  "residential_share": 0.33, "commercial_share": 0.36, "industrial_share": 0.32},
    {"zone": "DAYTON", "residential_share": 0.38, "commercial_share": 0.33, "industrial_share": 0.29},
    {"zone": "DUKE",   "residential_share": 0.38, "commercial_share": 0.38, "industrial_share": 0.24},
    {"zone": "VEPCO",  "residential_share": 0.34, "commercial_share": 0.56, "industrial_share": 0.10},
    {"zone": "DPL",    "residential_share": 0.48, "commercial_share": 0.37, "industrial_share": 0.15},
    {"zone": "DQE",    "residential_share": 0.33, "commercial_share": 0.46, "industrial_share": 0.21},
    {"zone": "EKPC",   "residential_share": 0.55, "commercial_share": 0.14, "industrial_share": 0.31},
    {"zone": "JCPL",   "residential_share": 0.49, "commercial_share": 0.42, "industrial_share": 0.09},
    {"zone": "METED",  "residential_share": 0.42, "commercial_share": 0.15, "industrial_share": 0.43},
    {"zone": "PECO",   "residential_share": 0.39, "commercial_share": 0.22, "industrial_share": 0.39},
    {"zone": "PEPCO",  "residential_share": 0.39, "commercial_share": 0.59, "industrial_share": 0.02},
    {"zone": "PL",     "residential_share": 0.40, "commercial_share": 0.38, "industrial_share": 0.23},
    {"zone": "PENLC",  "residential_share": 0.37, "commercial_share": 0.19, "industrial_share": 0.45},
    {"zone": "PS",     "residential_share": 0.35, "commercial_share": 0.56, "industrial_share": 0.09},
    {"zone": "RECO",   "residential_share": 0.49, "commercial_share": 0.50, "industrial_share": 0.01},
])

zone_load_forecast['forecast_datetime_ending_ept'] = pd.to_datetime(zone_load_forecast['forecast_datetime_ending_ept'])
zone_load_forecast['hour'] = zone_load_forecast['forecast_datetime_ending_ept'].dt.hour

df = zone_load_forecast.copy()

df = df.merge(hourly_sector_weight, on="hour", how="left")

df = df.merge(zone_sector_shares, on="zone", how="left")

# unnormalized
df["wtd_res"] = df["residential_share"] * df["w_res"]
df["wtd_com"]  = df["commercial_share"]  * df["w_com"]
df["wtd_ind"]  = df["industrial_share"]  * df["w_ind"]

# normalize
weight_sum = df[["wtd_res", "wtd_com", "wtd_ind"]].sum(axis=1)

df["res_forecast_load_mw"] = df["forecast_load_mw"] * df["wtd_res"] / weight_sum
df["com_forecast_load_mw"]  = df["forecast_load_mw"] * df["wtd_com"]  / weight_sum
df["ind_forecast_load_mw"]  = df["forecast_load_mw"] * df["wtd_ind"]  / weight_sum

zone_load_forecast = df.copy()

df = zonal_wcl.copy()

df = df.merge(hourly_sector_weight, on="hour", how="left")

df = df.merge(zone_sector_shares, on="zone", how="left")

# unnormalized
df["wtd_res"] = df["residential_share"] * df["w_res"]
df["wtd_com"]  = df["commercial_share"]  * df["w_com"]
df["wtd_ind"]  = df["industrial_share"]  * df["w_ind"]

# normalize
weight_sum = df[["wtd_res", "wtd_com", "wtd_ind"]].sum(axis=1)

df["res_MW"] = df["MW"] * df["wtd_res"] / weight_sum
df["com_MW"]  = df["MW"] * df["wtd_com"]  / weight_sum
df["ind_MW"]  = df["MW"] * df["wtd_ind"]  / weight_sum

zonal_wcl = df.copy()

res_zone_load = zonal_wcl[[
    'datetime_ending_ept', 'zone', 'res_MW',
    'temperature_2m', 'apparent_temperature', 'dew_point_2m', 
    'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'snow_depth', 
    'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'surface_pressure', 'pressure_msl', 'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'global_tilted_irradiance', 'direct_normal_irradiance', 'terrestrial_radiation',
    'date', 'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_holiday', 'WkDayBeforeHol', 'WkDayAfterHol', 'is_event'
]].copy()

com_zone_load = zonal_wcl[[
    'datetime_ending_ept', 'zone', 'com_MW',
    'temperature_2m', 'apparent_temperature', 'dew_point_2m', 
    'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'snow_depth', 
    'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'surface_pressure', 'pressure_msl', 'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'global_tilted_irradiance', 'direct_normal_irradiance', 'terrestrial_radiation',
    'date', 'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_holiday', 'WkDayBeforeHol', 'WkDayAfterHol', 'is_event'
]].copy()

ind_zone_load = zonal_wcl[[
    'datetime_ending_ept', 'zone', 'ind_MW',
    'temperature_2m', 'apparent_temperature', 'dew_point_2m', 
    'relative_humidity_2m', 'precipitation', 'rain', 'snowfall', 'snow_depth', 
    'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
    'surface_pressure', 'pressure_msl', 'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
    'shortwave_radiation', 'direct_radiation', 'diffuse_radiation', 'global_tilted_irradiance', 'direct_normal_irradiance', 'terrestrial_radiation',
    'date', 'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_holiday', 'WkDayBeforeHol', 'WkDayAfterHol', 'is_event'
]].copy()

############################################### residential ###################################################

alert = pd.read_csv("data/emergencymessages/alerts.csv")
alert["start"] = pd.to_datetime(alert["start"].astype(str).str[:-6])
alert["end"]   = pd.to_datetime(alert["end"].astype(str).str[:-6])

# make sure datetime types match
res_zone_load["datetime_ending_ept"] = pd.to_datetime(res_zone_load["datetime_ending_ept"])

# now your existing code works
res_zone_load["is_alert"] = 0

for _, row in alert.iterrows():
    mask = (
        (res_zone_load["datetime_ending_ept"] >= row["start"]) &
        (res_zone_load["datetime_ending_ept"] <= row["end"])
    )
    res_zone_load.loc[mask, "is_alert"] = 1

warning = pd.read_csv("data/emergencymessages/warnings.csv")
warning["start"] = pd.to_datetime(warning["start"].astype(str).str[:-6])
warning["end"]   = pd.to_datetime(warning["end"].astype(str).str[:-6])

res_zone_load["is_warning"] = 0

for _, row in warning.iterrows():
    mask = (
        (res_zone_load["datetime_ending_ept"] >= row["start"]) &
        (res_zone_load["datetime_ending_ept"] <= row["end"])
    )
    res_zone_load.loc[mask, "is_warning"] = 1

action = pd.read_csv("data/emergencymessages/actions.csv")
action["start"] = pd.to_datetime(action["start"].astype(str).str[:-6])
action["end"]   = pd.to_datetime(action["end"].astype(str).str[:-6])

res_zone_load["is_action"] = 0

for _, row in action.iterrows():
    mask = (
        (res_zone_load["datetime_ending_ept"] >= row["start"]) &
        (res_zone_load["datetime_ending_ept"] <= row["end"])
    )
    res_zone_load.loc[mask, "is_action"] = 1

# temp
daily_temp = (
    res_zone_load.groupby("date", as_index=False)["temperature_2m"]
    .agg(temp_min="min", temp_max="max")
)

res_zone_load = res_zone_load.merge(daily_temp, on="date", how="left")

res_zone_load["temp_f"] = res_zone_load["temperature_2m"] * 9/5 + 32

base_temp = 65

res_zone_load["HDD"] = np.maximum(base_temp - res_zone_load["temp_f"], 0)
res_zone_load["CDD"] = np.maximum(res_zone_load["temp_f"] - base_temp, 0)

res_zone_load["HDD_wind"] = res_zone_load["HDD"] * res_zone_load["wind_speed_10m"]
res_zone_load["CDD_cloud"] = res_zone_load["CDD"] * (1 - res_zone_load["cloud_cover"] / 100)

res_zone_load["wind_dir_10m_sin"] = np.sin(np.deg2rad(res_zone_load["wind_direction_10m"]))
res_zone_load["wind_dir_10m_cos"] = np.cos(np.deg2rad(res_zone_load["wind_direction_10m"]))

res_zone_load["wind_chill"] = (
    35.74
    + 0.6215 * res_zone_load["temp_f"]
    - 35.75 * (res_zone_load["wind_speed_10m"] ** 0.16)
    + 0.4275 * res_zone_load["temp_f"] * (res_zone_load["wind_speed_10m"] ** 0.16)
)

RH = res_zone_load["relative_humidity_2m"]
T = res_zone_load["temp_f"]

res_zone_load["heat_index_f"] = (
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

res_zone_load["feels_like_temp"] = np.where(
    (res_zone_load["temp_f"] <= 50) & (res_zone_load["wind_speed_10m"] > 3),
    res_zone_load["wind_chill"],
    np.where(
        (res_zone_load["temp_f"] >= 80) & (RH >= 40),
        res_zone_load["heat_index_f"],
        res_zone_load["temp_f"]
    )
)

res_zone_load = res_zone_load.sort_values(["zone", "datetime_ending_ept"]).copy()

# hour-to-hour absolute temperature change within each zone
res_zone_load["temp_diff_1h_abs"] = (
    res_zone_load.groupby("zone")["temperature_2m"]
    .diff()
    .abs()
)

# cumulative total variation over past 6 hours within each zone
res_zone_load["temp_total_variation_6h"] = (
    res_zone_load.groupby("zone")["temp_diff_1h_abs"]
    .transform(lambda s: s.rolling(6).sum())
)

# cumulative total variation over past 12 hours within each zone
res_zone_load["temp_total_variation_12h"] = (
    res_zone_load.groupby("zone")["temp_diff_1h_abs"]
    .transform(lambda s: s.rolling(12).sum())
)

# cumulative total variation over past 24 hours within each zone
res_zone_load["temp_total_variation_24h"] = (
    res_zone_load.groupby("zone")["temp_diff_1h_abs"]
    .transform(lambda s: s.rolling(24).sum())
)

# 1-hour ahead forecast temperature within each zone
res_zone_load["temp_fcst_1h"] = (
    res_zone_load.groupby("zone")["temperature_2m"]
    .shift(-1)
)

# 6-hour ahead mean forecast temperature within each zone
horizon = 6
res_zone_load["temp_lead_mean_6h"] = (
    sum(
        res_zone_load.groupby("zone")["temperature_2m"].shift(-i)
        for i in range(1, horizon + 1)
    ) / horizon
)

res_zone_load.drop(
    columns=["wind_chill", "temp_diff_1h_abs", "heat_index_f"],
    inplace=True
)

res_zone_load["datetime_ending_ept"] = pd.to_datetime(res_zone_load["datetime_ending_ept"])

def add_zone_features(df):
    df = df.sort_values("datetime_ending_ept").copy()

    # hour of target timestamp
    df["hour"] = df["datetime_ending_ept"].dt.hour

    # set day_shift
    df["day_shift"] = np.where(df["hour"].isin(range(1, 10)), 1, 2)

    # lookup index from historical MW for this zone only
    hist = (
        df[["datetime_ending_ept", "res_MW"]]
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

        df[time_col] = (
            df["datetime_ending_ept"]
            - pd.to_timedelta(df["day_shift"], unit="D")
            - pd.Timedelta(hours=lag)
        )

        df[feat_col] = hist["res_MW"].reindex(df[time_col]).to_numpy()

        helper_cols.append(time_col)
        feature_cols.append(feat_col)

    # rolling stats based only on historical series for this zone
    hist["MW_roll_24_base"] = hist["res_MW"].shift(1).rolling(24).mean()
    hist["MW_roll_48_base"] = hist["res_MW"].shift(1).rolling(48).mean()
    hist["MW_roll_72_base"] = hist["res_MW"].shift(1).rolling(72).mean()
    hist["MW_roll_168_base"] = hist["res_MW"].shift(1).rolling(168).mean()

    hist["MW_std_24_base"] = hist["res_MW"].shift(1).rolling(24).std()
    hist["MW_std_168_base"] = hist["res_MW"].shift(1).rolling(168).std()

    # rolling lookup time
    df["roll_time"] = (
        df["datetime_ending_ept"]
        - pd.to_timedelta(df["day_shift"], unit="D")
    )

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

    # drop rows without enough history
    df = df.dropna(subset=feature_cols).copy()

    # optional: remove helper columns
    df.drop(columns=["day_shift", "roll_time"] + helper_cols, inplace=True)

    return df

res_zone_load = (
    res_zone_load
    .groupby("zone")
    .apply(add_zone_features)
    .reset_index().drop(columns="level_1")
)


results = []
feat_imp_results = []

start_date = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() # pd.Timestamp("2026-03-25")
end_date   = start_date + pd.Timedelta(days=2) - pd.Timedelta(hours=1) # pd.Timestamp("2026-03-26")

zones = sorted(res_zone_load["zone"].dropna().unique())

for zone in zones:
    print(f"\n========== Running res zone: {zone} ==========")

    zone_df = (
        res_zone_load[res_zone_load["zone"] == zone]
        .sort_values("datetime_ending_ept")
        .copy()
    )

    zone_fcst = (
        zone_load_forecast[zone_load_forecast["zone"] == zone]
        .sort_values("forecast_datetime_ending_ept")
        .copy()
    )

    current_date = start_date

    while current_date <= end_date:

        print(f"Running for zone={zone}, date={current_date.date()}")

        # cutoff = previous day at 10:00
        cutoff = (current_date - pd.Timedelta(days=1)).replace(hour=10)

        # train set: only information available by cutoff
        train_df = zone_df[
            zone_df["datetime_ending_ept"] <= cutoff
        ].copy()

        # trimming: compute thresholds from training data only
        lower = train_df["res_MW"].quantile(0.01)
        upper = train_df["res_MW"].quantile(0.99)

        # trim training set
        train_df = train_df[
            (train_df["res_MW"] >= lower) &
            (train_df["res_MW"] <= upper)
        ].copy()


        # test set: full target day
        test_start = current_date.replace(hour=1)
        test_end   = current_date.replace(hour=23) + pd.Timedelta(hours=1)

        test_df = zone_df[
            (zone_df["datetime_ending_ept"] >= test_start) &
            (zone_df["datetime_ending_ept"] <= test_end)
        ].copy()

        # skip if empty
        if train_df.empty or test_df.empty:
            print("Skipping because train/test is empty")
            current_date += pd.Timedelta(days=1)
            continue

        target = "res_MW"

        drop_cols = ["res_MW", "datetime_ending_ept", "zone", "date"]
        features = [col for col in train_df.columns if col not in drop_cols]

        # time-based validation split from training data
        # use the most recent 14 days of hourly data as validation
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

        # feature importance for this day + zone
        feat_imp = pd.DataFrame({
            "zone": zone,
            "date": current_date,
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        feat_imp["rank"] = range(1, len(feat_imp) + 1)
        feat_imp_results.append(feat_imp)

        test_df = test_df.copy()
        test_df["res_MW_pred"] = model.predict(X_test)

        # zonal forecast for same target day
        forecast_df = (
            zone_fcst[
                (zone_fcst["forecast_datetime_ending_ept"] >= test_start) &
                (zone_fcst["forecast_datetime_ending_ept"] <= test_end)
            ]
            .sort_values("forecast_datetime_ending_ept")
            .copy()
        )

        forecast_df = forecast_df.rename(columns={
            "forecast_datetime_ending_ept": "datetime_ending_ept"
        })

        forecast_df["datetime_ending_ept"] = pd.to_datetime(
            forecast_df["datetime_ending_ept"]
        )

        compare_df = pd.merge(
            test_df,
            forecast_df[["datetime_ending_ept", "res_forecast_load_mw"]],
            on="datetime_ending_ept",
            how="inner"
        )

        if compare_df.empty:
            print("Skipping because compare_df is empty after forecast merge")
            current_date += pd.Timedelta(days=1)
            continue

        mae = mean_absolute_error(compare_df["res_forecast_load_mw"], compare_df["res_MW_pred"])
        rmse = np.sqrt(mean_squared_error(compare_df["res_forecast_load_mw"], compare_df["res_MW_pred"]))

        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        compare_df["zone"] = zone
        compare_df["date"] = current_date
        compare_df["MAE"] = mae
        compare_df["RMSE"] = rmse

        results.append(compare_df)

        current_date += pd.Timedelta(days=1)

# combine all days / zones
res_final = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
res_final_feat_importance = pd.concat(feat_imp_results, ignore_index=True) if feat_imp_results else pd.DataFrame()

##################################### commercial ##############################################
# make sure datetime types match
com_zone_load["datetime_ending_ept"] = pd.to_datetime(com_zone_load["datetime_ending_ept"])

# alert 
com_zone_load["is_alert"] = 0

for _, row in alert.iterrows():
    mask = (
        (com_zone_load["datetime_ending_ept"] >= row["start"]) &
        (com_zone_load["datetime_ending_ept"] <= row["end"])
    )
    com_zone_load.loc[mask, "is_alert"] = 1

com_zone_load["is_warning"] = 0

for _, row in warning.iterrows():
    mask = (
        (com_zone_load["datetime_ending_ept"] >= row["start"]) &
        (com_zone_load["datetime_ending_ept"] <= row["end"])
    )
    com_zone_load.loc[mask, "is_warning"] = 1

com_zone_load["is_action"] = 0

for _, row in action.iterrows():
    mask = (
        (com_zone_load["datetime_ending_ept"] >= row["start"]) &
        (com_zone_load["datetime_ending_ept"] <= row["end"])
    )
    com_zone_load.loc[mask, "is_action"] = 1

# temp
daily_temp = (
    com_zone_load.groupby("date", as_index=False)["temperature_2m"]
    .agg(temp_min="min", temp_max="max")
)

com_zone_load = com_zone_load.merge(daily_temp, on="date", how="left")

com_zone_load["temp_f"] = com_zone_load["temperature_2m"] * 9/5 + 32

base_temp = 65

com_zone_load["HDD"] = np.maximum(base_temp - com_zone_load["temp_f"], 0)
com_zone_load["CDD"] = np.maximum(com_zone_load["temp_f"] - base_temp, 0)

com_zone_load["HDD_wind"] = com_zone_load["HDD"] * com_zone_load["wind_speed_10m"]
com_zone_load["CDD_cloud"] = com_zone_load["CDD"] * (1 - com_zone_load["cloud_cover"] / 100)

com_zone_load["wind_dir_10m_sin"] = np.sin(np.deg2rad(com_zone_load["wind_direction_10m"]))
com_zone_load["wind_dir_10m_cos"] = np.cos(np.deg2rad(com_zone_load["wind_direction_10m"]))

com_zone_load["wind_chill"] = (
    35.74
    + 0.6215 * com_zone_load["temp_f"]
    - 35.75 * (com_zone_load["wind_speed_10m"] ** 0.16)
    + 0.4275 * com_zone_load["temp_f"] * (com_zone_load["wind_speed_10m"] ** 0.16)
)

RH = com_zone_load["relative_humidity_2m"]
T = com_zone_load["temp_f"]

com_zone_load["heat_index_f"] = (
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

com_zone_load["feels_like_temp"] = np.where(
    (com_zone_load["temp_f"] <= 50) & (com_zone_load["wind_speed_10m"] > 3),
    com_zone_load["wind_chill"],
    np.where(
        (com_zone_load["temp_f"] >= 80) & (RH >= 40),
        com_zone_load["heat_index_f"],
        com_zone_load["temp_f"]
    )
)

com_zone_load = com_zone_load.sort_values(["zone", "datetime_ending_ept"]).copy()

# hour-to-hour absolute temperature change within each zone
com_zone_load["temp_diff_1h_abs"] = (
    com_zone_load.groupby("zone")["temperature_2m"]
    .diff()
    .abs()
)

# cumulative total variation over past 6 hours within each zone
com_zone_load["temp_total_variation_6h"] = (
    com_zone_load.groupby("zone")["temp_diff_1h_abs"]
    .transform(lambda s: s.rolling(6).sum())
)

# cumulative total variation over past 12 hours within each zone
com_zone_load["temp_total_variation_12h"] = (
    com_zone_load.groupby("zone")["temp_diff_1h_abs"]
    .transform(lambda s: s.rolling(12).sum())
)

# cumulative total variation over past 24 hours within each zone
com_zone_load["temp_total_variation_24h"] = (
    com_zone_load.groupby("zone")["temp_diff_1h_abs"]
    .transform(lambda s: s.rolling(24).sum())
)

# 1-hour ahead forecast temperature within each zone
com_zone_load["temp_fcst_1h"] = (
    com_zone_load.groupby("zone")["temperature_2m"]
    .shift(-1)
)

# 6-hour ahead mean forecast temperature within each zone
horizon = 6
com_zone_load["temp_lead_mean_6h"] = (
    sum(
        com_zone_load.groupby("zone")["temperature_2m"].shift(-i)
        for i in range(1, horizon + 1)
    ) / horizon
)

com_zone_load.drop(
    columns=["wind_chill", "temp_diff_1h_abs", "heat_index_f"],
    inplace=True
)

com_zone_load["datetime_ending_ept"] = pd.to_datetime(com_zone_load["datetime_ending_ept"])

def add_zone_features(df):
    df = df.sort_values("datetime_ending_ept").copy()

    # hour of target timestamp
    df["hour"] = df["datetime_ending_ept"].dt.hour

    # set day_shift
    df["day_shift"] = np.where(df["hour"].isin(range(1, 10)), 1, 2)

    # lookup index from historical MW for this zone only
    hist = (
        df[["datetime_ending_ept", "com_MW"]]
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

        df[time_col] = (
            df["datetime_ending_ept"]
            - pd.to_timedelta(df["day_shift"], unit="D")
            - pd.Timedelta(hours=lag)
        )

        df[feat_col] = hist["com_MW"].reindex(df[time_col]).to_numpy()

        helper_cols.append(time_col)
        feature_cols.append(feat_col)

    # rolling stats based only on historical series for this zone
    hist["MW_roll_24_base"] = hist["com_MW"].shift(1).rolling(24).mean()
    hist["MW_roll_48_base"] = hist["com_MW"].shift(1).rolling(48).mean()
    hist["MW_roll_72_base"] = hist["com_MW"].shift(1).rolling(72).mean()
    hist["MW_roll_168_base"] = hist["com_MW"].shift(1).rolling(168).mean()

    hist["MW_std_24_base"] = hist["com_MW"].shift(1).rolling(24).std()
    hist["MW_std_168_base"] = hist["com_MW"].shift(1).rolling(168).std()

    # rolling lookup time
    df["roll_time"] = (
        df["datetime_ending_ept"]
        - pd.to_timedelta(df["day_shift"], unit="D")
    )

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

    # drop rows without enough history
    df = df.dropna(subset=feature_cols).copy()

    # optional: remove helper columns
    df.drop(columns=["day_shift", "roll_time"] + helper_cols, inplace=True)

    return df

com_zone_load = (
    com_zone_load
    .groupby("zone")
    .apply(add_zone_features)
    .reset_index().drop(columns="level_1")
)


results = []
feat_imp_results = []

start_date = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() # pd.Timestamp("2026-03-25")
end_date   = start_date + pd.Timedelta(days=2) - pd.Timedelta(hours=1) # pd.Timestamp("2026-03-26")

zones = sorted(com_zone_load["zone"].dropna().unique())

for zone in zones:
    print(f"\n========== Running com zone: {zone} ==========")

    zone_df = (
        com_zone_load[com_zone_load["zone"] == zone]
        .sort_values("datetime_ending_ept")
        .copy()
    )

    zone_fcst = (
        zone_load_forecast[zone_load_forecast["zone"] == zone]
        .sort_values("forecast_datetime_ending_ept")
        .copy()
    )

    current_date = start_date

    while current_date <= end_date:

        print(f"Running for zone={zone}, date={current_date.date()}")

        # cutoff = previous day at 10:00
        cutoff = (current_date - pd.Timedelta(days=1)).replace(hour=10)

        # train set: only information available by cutoff
        train_df = zone_df[
            zone_df["datetime_ending_ept"] <= cutoff
        ].copy()

        # trimming: compute thresholds from training data only
        lower = train_df["com_MW"].quantile(0.01)
        upper = train_df["com_MW"].quantile(0.99)

        # trim training set
        train_df = train_df[
            (train_df["com_MW"] >= lower) &
            (train_df["com_MW"] <= upper)
        ].copy()


        # test set: full target day
        test_start = current_date.replace(hour=1)
        test_end   = current_date.replace(hour=23) + pd.Timedelta(hours=1)

        test_df = zone_df[
            (zone_df["datetime_ending_ept"] >= test_start) &
            (zone_df["datetime_ending_ept"] <= test_end)
        ].copy()

        # skip if empty
        if train_df.empty or test_df.empty:
            print("Skipping because train/test is empty")
            current_date += pd.Timedelta(days=1)
            continue

        target = "com_MW"

        drop_cols = ["com_MW", "datetime_ending_ept", "zone", "date"]
        features = [col for col in train_df.columns if col not in drop_cols]

        # time-based validation split from training data
        # use the most recent 14 days of hourly data as validation
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

        # feature importance for this day + zone
        feat_imp = pd.DataFrame({
            "zone": zone,
            "date": current_date,
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        feat_imp["rank"] = range(1, len(feat_imp) + 1)
        feat_imp_results.append(feat_imp)

        test_df = test_df.copy()
        test_df["com_MW_pred"] = model.predict(X_test)

        # zonal forecast for same target day
        forecast_df = (
            zone_fcst[
                (zone_fcst["forecast_datetime_ending_ept"] >= test_start) &
                (zone_fcst["forecast_datetime_ending_ept"] <= test_end)
            ]
            .sort_values("forecast_datetime_ending_ept")
            .copy()
        )

        forecast_df = forecast_df.rename(columns={
            "forecast_datetime_ending_ept": "datetime_ending_ept"
        })

        forecast_df["datetime_ending_ept"] = pd.to_datetime(
            forecast_df["datetime_ending_ept"]
        )

        compare_df = pd.merge(
            test_df,
            forecast_df[["datetime_ending_ept", "com_forecast_load_mw"]],
            on="datetime_ending_ept",
            how="inner"
        )

        if compare_df.empty:
            print("Skipping because compare_df is empty after forecast merge")
            current_date += pd.Timedelta(days=1)
            continue

        mae = mean_absolute_error(compare_df["com_forecast_load_mw"], compare_df["com_MW_pred"])
        rmse = np.sqrt(mean_squared_error(compare_df["com_forecast_load_mw"], compare_df["com_MW_pred"]))

        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        compare_df["zone"] = zone
        compare_df["date"] = current_date
        compare_df["MAE"] = mae
        compare_df["RMSE"] = rmse

        results.append(compare_df)

        current_date += pd.Timedelta(days=1)

# combine all days / zones
com_final = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
com_final_feat_importance = pd.concat(feat_imp_results, ignore_index=True) if feat_imp_results else pd.DataFrame()

########################## industrial ##############################################

ind_zone_load["datetime_ending_ept"] = pd.to_datetime(ind_zone_load["datetime_ending_ept"])

ind_zone_load["is_alert"] = 0

for _, row in alert.iterrows():
    mask = (
        (ind_zone_load["datetime_ending_ept"] >= row["start"]) &
        (ind_zone_load["datetime_ending_ept"] <= row["end"])
    )
    ind_zone_load.loc[mask, "is_alert"] = 1

ind_zone_load["is_warning"] = 0

for _, row in warning.iterrows():
    mask = (
        (ind_zone_load["datetime_ending_ept"] >= row["start"]) &
        (ind_zone_load["datetime_ending_ept"] <= row["end"])
    )
    ind_zone_load.loc[mask, "is_warning"] = 1

ind_zone_load["is_action"] = 0

for _, row in action.iterrows():
    mask = (
        (ind_zone_load["datetime_ending_ept"] >= row["start"]) &
        (ind_zone_load["datetime_ending_ept"] <= row["end"])
    )
    ind_zone_load.loc[mask, "is_action"] = 1

# temp
daily_temp = (
    ind_zone_load.groupby("date", as_index=False)["temperature_2m"]
    .agg(temp_min="min", temp_max="max")
)

ind_zone_load = ind_zone_load.merge(daily_temp, on="date", how="left")

ind_zone_load["temp_f"] = ind_zone_load["temperature_2m"] * 9/5 + 32

base_temp = 65

ind_zone_load["HDD"] = np.maximum(base_temp - ind_zone_load["temp_f"], 0)
ind_zone_load["CDD"] = np.maximum(ind_zone_load["temp_f"] - base_temp, 0)

ind_zone_load["HDD_wind"] = ind_zone_load["HDD"] * ind_zone_load["wind_speed_10m"]
ind_zone_load["CDD_cloud"] = ind_zone_load["CDD"] * (1 - ind_zone_load["cloud_cover"] / 100)

ind_zone_load["wind_dir_10m_sin"] = np.sin(np.deg2rad(ind_zone_load["wind_direction_10m"]))
ind_zone_load["wind_dir_10m_cos"] = np.cos(np.deg2rad(ind_zone_load["wind_direction_10m"]))

ind_zone_load["wind_chill"] = (
    35.74
    + 0.6215 * ind_zone_load["temp_f"]
    - 35.75 * (ind_zone_load["wind_speed_10m"] ** 0.16)
    + 0.4275 * ind_zone_load["temp_f"] * (ind_zone_load["wind_speed_10m"] ** 0.16)
)

RH = ind_zone_load["relative_humidity_2m"]
T = ind_zone_load["temp_f"]

ind_zone_load["heat_index_f"] = (
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

ind_zone_load["feels_like_temp"] = np.where(
    (ind_zone_load["temp_f"] <= 50) & (ind_zone_load["wind_speed_10m"] > 3),
    ind_zone_load["wind_chill"],
    np.where(
        (ind_zone_load["temp_f"] >= 80) & (RH >= 40),
        ind_zone_load["heat_index_f"],
        ind_zone_load["temp_f"]
    )
)

ind_zone_load = ind_zone_load.sort_values(["zone", "datetime_ending_ept"]).copy()

# hour-to-hour absolute temperature change within each zone
ind_zone_load["temp_diff_1h_abs"] = (
    ind_zone_load.groupby("zone")["temperature_2m"]
    .diff()
    .abs()
)

# cumulative total variation over past 6 hours within each zone
ind_zone_load["temp_total_variation_6h"] = (
    ind_zone_load.groupby("zone")["temp_diff_1h_abs"]
    .transform(lambda s: s.rolling(6).sum())
)

# cumulative total variation over past 12 hours within each zone
ind_zone_load["temp_total_variation_12h"] = (
    ind_zone_load.groupby("zone")["temp_diff_1h_abs"]
    .transform(lambda s: s.rolling(12).sum())
)

# cumulative total variation over past 24 hours within each zone
ind_zone_load["temp_total_variation_24h"] = (
    ind_zone_load.groupby("zone")["temp_diff_1h_abs"]
    .transform(lambda s: s.rolling(24).sum())
)

# 1-hour ahead forecast temperature within each zone
ind_zone_load["temp_fcst_1h"] = (
    ind_zone_load.groupby("zone")["temperature_2m"]
    .shift(-1)
)

# 6-hour ahead mean forecast temperature within each zone
horizon = 6
ind_zone_load["temp_lead_mean_6h"] = (
    sum(
        ind_zone_load.groupby("zone")["temperature_2m"].shift(-i)
        for i in range(1, horizon + 1)
    ) / horizon
)

ind_zone_load.drop(
    columns=["wind_chill", "temp_diff_1h_abs", "heat_index_f"],
    inplace=True
)

ind_zone_load["datetime_ending_ept"] = pd.to_datetime(ind_zone_load["datetime_ending_ept"])

def add_zone_features(df):
    df = df.sort_values("datetime_ending_ept").copy()

    # hour of target timestamp
    df["hour"] = df["datetime_ending_ept"].dt.hour

    # set day_shift
    df["day_shift"] = np.where(df["hour"].isin(range(1, 10)), 1, 2)

    # lookup index from historical MW for this zone only
    hist = (
        df[["datetime_ending_ept", "ind_MW"]]
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

        df[time_col] = (
            df["datetime_ending_ept"]
            - pd.to_timedelta(df["day_shift"], unit="D")
            - pd.Timedelta(hours=lag)
        )

        df[feat_col] = hist["ind_MW"].reindex(df[time_col]).to_numpy()

        helper_cols.append(time_col)
        feature_cols.append(feat_col)

    # rolling stats based only on historical series for this zone
    hist["MW_roll_24_base"] = hist["ind_MW"].shift(1).rolling(24).mean()
    hist["MW_roll_48_base"] = hist["ind_MW"].shift(1).rolling(48).mean()
    hist["MW_roll_72_base"] = hist["ind_MW"].shift(1).rolling(72).mean()
    hist["MW_roll_168_base"] = hist["ind_MW"].shift(1).rolling(168).mean()

    hist["MW_std_24_base"] = hist["ind_MW"].shift(1).rolling(24).std()
    hist["MW_std_168_base"] = hist["ind_MW"].shift(1).rolling(168).std()

    # rolling lookup time
    df["roll_time"] = (
        df["datetime_ending_ept"]
        - pd.to_timedelta(df["day_shift"], unit="D")
    )

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

    # drop rows without enough history
    df = df.dropna(subset=feature_cols).copy()

    # optional: remove helper columns
    df.drop(columns=["day_shift", "roll_time"] + helper_cols, inplace=True)

    return df

ind_zone_load = (
    ind_zone_load
    .groupby("zone")
    .apply(add_zone_features)
    .reset_index().drop(columns="level_1")
)


results = []
feat_imp_results = []

start_date = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() # pd.Timestamp("2026-03-25")
end_date   = start_date + pd.Timedelta(days=2) - pd.Timedelta(hours=1) # pd.Timestamp("2026-03-26")

zones = sorted(ind_zone_load["zone"].dropna().unique())

for zone in zones:
    print(f"\n========== Running ind zone: {zone} ==========")

    zone_df = (
        ind_zone_load[ind_zone_load["zone"] == zone]
        .sort_values("datetime_ending_ept")
        .copy()
    )

    zone_fcst = (
        zone_load_forecast[zone_load_forecast["zone"] == zone]
        .sort_values("forecast_datetime_ending_ept")
        .copy()
    )

    current_date = start_date

    while current_date <= end_date:

        print(f"Running for zone={zone}, date={current_date.date()}")

        # cutoff = previous day at 10:00
        cutoff = (current_date - pd.Timedelta(days=1)).replace(hour=10)

        # train set: only information available by cutoff
        train_df = zone_df[
            zone_df["datetime_ending_ept"] <= cutoff
        ].copy()

        # trimming: compute thresholds from training data only
        lower = train_df["ind_MW"].quantile(0.01)
        upper = train_df["ind_MW"].quantile(0.99)

        # trim training set
        train_df = train_df[
            (train_df["ind_MW"] >= lower) &
            (train_df["ind_MW"] <= upper)
        ].copy()


        # test set: full target day
        test_start = current_date.replace(hour=1)
        test_end   = current_date.replace(hour=23) + pd.Timedelta(hours=1)

        test_df = zone_df[
            (zone_df["datetime_ending_ept"] >= test_start) &
            (zone_df["datetime_ending_ept"] <= test_end)
        ].copy()

        # skip if empty
        if train_df.empty or test_df.empty:
            print("Skipping because train/test is empty")
            current_date += pd.Timedelta(days=1)
            continue

        target = "ind_MW"

        drop_cols = ["ind_MW", "datetime_ending_ept", "zone", "date"]
        features = [col for col in train_df.columns if col not in drop_cols]

        # time-based validation split from training data
        # use the most recent 14 days of hourly data as validation
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

        # feature importance for this day + zone
        feat_imp = pd.DataFrame({
            "zone": zone,
            "date": current_date,
            "feature": X_train.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        feat_imp["rank"] = range(1, len(feat_imp) + 1)
        feat_imp_results.append(feat_imp)

        test_df = test_df.copy()
        test_df["ind_MW_pred"] = model.predict(X_test)

        # zonal forecast for same target day
        forecast_df = (
            zone_fcst[
                (zone_fcst["forecast_datetime_ending_ept"] >= test_start) &
                (zone_fcst["forecast_datetime_ending_ept"] <= test_end)
            ]
            .sort_values("forecast_datetime_ending_ept")
            .copy()
        )

        forecast_df = forecast_df.rename(columns={
            "forecast_datetime_ending_ept": "datetime_ending_ept"
        })

        forecast_df["datetime_ending_ept"] = pd.to_datetime(
            forecast_df["datetime_ending_ept"]
        )

        compare_df = pd.merge(
            test_df,
            forecast_df[["datetime_ending_ept", "ind_forecast_load_mw"]],
            on="datetime_ending_ept",
            how="inner"
        )

        if compare_df.empty:
            print("Skipping because compare_df is empty after forecast merge")
            current_date += pd.Timedelta(days=1)
            continue

        mae = mean_absolute_error(compare_df["ind_forecast_load_mw"], compare_df["ind_MW_pred"])
        rmse = np.sqrt(mean_squared_error(compare_df["ind_forecast_load_mw"], compare_df["ind_MW_pred"]))

        print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

        compare_df["zone"] = zone
        compare_df["date"] = current_date
        compare_df["MAE"] = mae
        compare_df["RMSE"] = rmse

        results.append(compare_df)

        current_date += pd.Timedelta(days=1)

# combine all days / zones
ind_final = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
ind_final_feat_importance = pd.concat(feat_imp_results, ignore_index=True) if feat_imp_results else pd.DataFrame()

############################################## sector union ###################################################

zone_load_forecast.rename(columns={"forecast_datetime_ending_ept": "datetime_ending_ept"}, inplace=True)

# merge predictions together
pred_df = (
    res_final[["datetime_ending_ept", "zone", "res_MW_pred"]]
    .merge(
        com_final[["datetime_ending_ept", "zone", "com_MW_pred"]],
        on=["datetime_ending_ept", "zone"],
        how="outer"
    )
    .merge(
        ind_final[["datetime_ending_ept", "zone", "ind_MW_pred"]],
        on=["datetime_ending_ept", "zone"],
        how="outer"
    )
)

# total prediction per zone
pred_df["total_MW_pred"] = (
    pred_df["res_MW_pred"].fillna(0)
    + pred_df["com_MW_pred"].fillna(0)
    + pred_df["ind_MW_pred"].fillna(0)
)

# merge actual load (must also be zonal or compatible)
pred_df = (
    pred_df
    .merge(
        zonal_wcl[["datetime_ending_ept", "zone", "MW"]],
        on=["datetime_ending_ept", "zone"],
        how="left"
    )
)

# merge forecast
pred_df = (
    pred_df
    .merge(
        zone_load_forecast[["datetime_ending_ept", "zone", "forecast_load_mw"]],
        on=["datetime_ending_ept", "zone"],
        how="left"
    )
)

# extract date
pred_df["date"] = pred_df["datetime_ending_ept"].dt.date
pred_df["hour"] = pred_df["datetime_ending_ept"].dt.hour

today = datetime.today() - timedelta(days=offset)

start_str = today.strftime("%m%d")                 # 0325
end_str = (today + timedelta(days=1)).strftime("%m%d")  # 0326
run_str = today.strftime("%y%m%d")                 # 260325

pred_df[
    ['datetime_ending_ept', 'zone', 'date', 'hour', 'total_MW_pred', 'forecast_load_mw']
].to_csv(
    f'data/prediction/zone_sector_prediction_{start_str}_to_{end_str}_at_{run_str}.csv',
    index=False
)

