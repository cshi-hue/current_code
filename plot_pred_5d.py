import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from datetime import datetime, timedelta

offset = 0
today = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize()

d0 = today
d1 = today - pd.Timedelta(days=1)
d2 = today - pd.Timedelta(days=2)
d3 = today - pd.Timedelta(days=3)
d4 = today - pd.Timedelta(days=4)

# format strings
fmt_mmdd = lambda d: d.strftime("%m%d")
fmt_yymmdd = lambda d: d.strftime("%y%m%d")

# RTO prediction 
rto_pred_0 = pd.read_csv(
    f"data/prediction/RTO_prediction_{fmt_mmdd(d0)}_to_{fmt_mmdd(d0 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d0)}.csv"
)
rto_pred_1 = pd.read_csv(
    f"data/prediction/RTO_prediction_{fmt_mmdd(d1)}_to_{fmt_mmdd(d1 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d1)}.csv"
)
rto_pred_2 = pd.read_csv(
    f"data/prediction/RTO_prediction_{fmt_mmdd(d2)}_to_{fmt_mmdd(d2 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d2)}.csv"
)
rto_pred_3 = pd.read_csv(
    f"data/prediction/RTO_prediction_{fmt_mmdd(d3)}_to_{fmt_mmdd(d3 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d3)}.csv"
)
rto_pred_4 = pd.read_csv(
    f"data/prediction/RTO_prediction_{fmt_mmdd(d4)}_to_{fmt_mmdd(d4 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d4)}.csv"
)

rto_pred_0["datetime_ending_ept"] = pd.to_datetime(rto_pred_0["datetime_ending_ept"])
rto_pred_1["datetime_ending_ept"] = pd.to_datetime(rto_pred_1["datetime_ending_ept"])
rto_pred_2["datetime_ending_ept"] = pd.to_datetime(rto_pred_2["datetime_ending_ept"])
rto_pred_3["datetime_ending_ept"] = pd.to_datetime(rto_pred_3["datetime_ending_ept"])
rto_pred_4["datetime_ending_ept"] = pd.to_datetime(rto_pred_4["datetime_ending_ept"])


# RTO sector prediction 
rto_sector_0 = pd.read_csv(
    f"data/prediction/RTO_sector_prediction_{fmt_mmdd(d0)}_to_{fmt_mmdd(d0 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d0)}.csv"
)
rto_sector_1 = pd.read_csv(
    f"data/prediction/RTO_sector_prediction_{fmt_mmdd(d1)}_to_{fmt_mmdd(d1 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d1)}.csv"
)
rto_sector_2 = pd.read_csv(
    f"data/prediction/RTO_sector_prediction_{fmt_mmdd(d2)}_to_{fmt_mmdd(d2 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d2)}.csv"
)
rto_sector_3 = pd.read_csv(
    f"data/prediction/RTO_sector_prediction_{fmt_mmdd(d3)}_to_{fmt_mmdd(d3 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d3)}.csv"
)
rto_sector_4 = pd.read_csv(
    f"data/prediction/RTO_sector_prediction_{fmt_mmdd(d4)}_to_{fmt_mmdd(d4 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d4)}.csv"
)

rto_sector_0["datetime_ending_ept"] = pd.to_datetime(rto_sector_0["datetime_ending_ept"])
rto_sector_1["datetime_ending_ept"] = pd.to_datetime(rto_sector_1["datetime_ending_ept"])
rto_sector_2["datetime_ending_ept"] = pd.to_datetime(rto_sector_2["datetime_ending_ept"])
rto_sector_3["datetime_ending_ept"] = pd.to_datetime(rto_sector_3["datetime_ending_ept"])
rto_sector_4["datetime_ending_ept"] = pd.to_datetime(rto_sector_4["datetime_ending_ept"])

######### zone aggregation ###################
zone_pred_4 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d4)}_to_{fmt_mmdd(d4 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d4)}.csv")
zone_pred_3 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d3)}_to_{fmt_mmdd(d3 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d3)}.csv")
zone_pred_2 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d2)}_to_{fmt_mmdd(d2 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d2)}.csv")
zone_pred_1 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d1)}_to_{fmt_mmdd(d1 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d1)}.csv")
zone_pred_0 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d0)}_to_{fmt_mmdd(d0 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d0)}.csv")

zone_pred_4["datetime_ending_ept"] = pd.to_datetime(zone_pred_4["datetime_ending_ept"])
zone_pred_3["datetime_ending_ept"] = pd.to_datetime(zone_pred_3["datetime_ending_ept"])
zone_pred_2["datetime_ending_ept"] = pd.to_datetime(zone_pred_2["datetime_ending_ept"])
zone_pred_1["datetime_ending_ept"] = pd.to_datetime(zone_pred_1["datetime_ending_ept"])
zone_pred_0["datetime_ending_ept"] = pd.to_datetime(zone_pred_0["datetime_ending_ept"])

zone_sector_4 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d4)}_to_{fmt_mmdd(d4 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d4)}.csv")
zone_sector_3 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d3)}_to_{fmt_mmdd(d3 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d3)}.csv")
zone_sector_2 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d2)}_to_{fmt_mmdd(d2 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d2)}.csv")
zone_sector_1 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d1)}_to_{fmt_mmdd(d1 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d1)}.csv")
zone_sector_0 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d0)}_to_{fmt_mmdd(d0 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d0)}.csv")

zone_sector_4["datetime_ending_ept"] = pd.to_datetime(zone_sector_4["datetime_ending_ept"])
zone_sector_3["datetime_ending_ept"] = pd.to_datetime(zone_sector_3["datetime_ending_ept"])
zone_sector_2["datetime_ending_ept"] = pd.to_datetime(zone_sector_2["datetime_ending_ept"])
zone_sector_1["datetime_ending_ept"] = pd.to_datetime(zone_sector_1["datetime_ending_ept"])
zone_sector_0["datetime_ending_ept"] = pd.to_datetime(zone_sector_0["datetime_ending_ept"])

# RTO total forecast load by datetime
rto_zone_pred = (
    zone_pred_0
    .groupby("datetime_ending_ept", as_index=False)["MW_pred"]
    .sum()
)

rto_zone_sector = (
    zone_sector_0
    .groupby("datetime_ending_ept", as_index=False)["total_MW_pred"]
    .sum()
)

#####################################################

instant_load_0 = pd.read_csv(f"data/pjm_All_Instantaneous_Load_rt5/pjm_All_Instantaneous_Load_rt5_{fmt_mmdd(d0)}.csv", skiprows = [1])

instant_load_0['instantaneous_load'] = pd.to_numeric(instant_load_0['instantaneous_load'], errors='coerce')

instant_load_0["datetime_beginning_ept"] = pd.to_datetime(
    instant_load_0["datetime_beginning_ept"]
)
    
hourly_load_0 = (
    instant_load_0
    .assign(
        hour_beginning_ept=lambda x: x["datetime_beginning_ept"].dt.floor("h"),
        datetime_ending_ept=lambda x: x["datetime_beginning_ept"].dt.floor("h") + pd.Timedelta(hours=1)
    )
    .groupby(["area", "datetime_ending_ept"], as_index=False)["instantaneous_load"]
    .mean()
    .rename(columns={"instantaneous_load": "load_mw_hourly_avg"})
)

actual_pjm = (
    hourly_load_0[
        (hourly_load_0["area"] == "PJM RTO") &
        (hourly_load_0["datetime_ending_ept"] >= ((pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() - pd.Timedelta(days=3)))
    ]
    .sort_values("datetime_ending_ept")
    .copy()
)


today = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize()


cut_d_1_start = today - pd.Timedelta(days=3)                   # T-3 00:00:00

cut_d_1_end = today - pd.Timedelta(days=2) - pd.Timedelta(seconds=1)  # T-3 23:59:59
cut_d0_start = today - pd.Timedelta(days=2)                   # T-2 00:00:00

cut_d0_end = today - -pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # T-1 23:59:59
cut_d1_start = today - pd.Timedelta(days=1)                   # T-1 00:00:00    

cut_d1_end = today - pd.Timedelta(seconds=1)                  # T-1 23:59:59
cut_d2_start = today                                          # T 00:00:00

cut_d2_end = today + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # T 23:59:59
cut_d3_start = today + pd.Timedelta(days=1)                   # T+1 00:00:00

forecast_part_1 = rto_pred_4[
    (rto_pred_4["datetime_ending_ept"] >= cut_d_1_start) &
    (rto_pred_4["datetime_ending_ept"] <= cut_d_1_end)
][["datetime_ending_ept", "forecast_load_mw"]]

forecast_part0 = rto_pred_3[
    (rto_pred_3["datetime_ending_ept"] >= cut_d0_start) & 
    (rto_pred_3["datetime_ending_ept"] <= cut_d0_end)
][["datetime_ending_ept", "forecast_load_mw"]]

forecast_part1 = rto_pred_2[
    (rto_pred_2["datetime_ending_ept"] >= cut_d1_start) &
    (rto_pred_2["datetime_ending_ept"] <= cut_d1_end)
][["datetime_ending_ept", "forecast_load_mw"]]

forecast_part2 = rto_pred_1[
    (rto_pred_1["datetime_ending_ept"] >= cut_d2_start) &
    (rto_pred_1["datetime_ending_ept"] <= cut_d2_end)
][["datetime_ending_ept", "forecast_load_mw"]]

forecast_part3 = rto_pred_0[
    rto_pred_0["datetime_ending_ept"] >= cut_d3_start
][["datetime_ending_ept", "forecast_load_mw"]]

forecast_combined = pd.concat([forecast_part_1, forecast_part0, forecast_part1, forecast_part2, forecast_part3])

rto_pred_4 = rto_pred_4[rto_pred_4["datetime_ending_ept"] >= cut_d_1_start].copy()
rto_sector_4 = rto_sector_4[rto_sector_4["datetime_ending_ept"] >= cut_d_1_start].copy()

sm_fct = 0.5  # Adjust this value for more or less smoothing

fig = go.Figure()

# Line 1: stitched forecast
fig.add_trace(go.Scatter(
    x=forecast_combined["datetime_ending_ept"],
    y=forecast_combined["forecast_load_mw"],
    mode="lines",
    name="PJM Forecast",
    line=dict(width=3, shape="spline", smoothing=sm_fct, color="#000000")  # black
))

# Line 2: MW_pred 
fig.add_trace(go.Scatter(
    x=rto_pred_4["datetime_ending_ept"],
    y=rto_pred_4["MW_pred"],
    mode="lines",
    name=f"MW_pred ({fmt_mmdd(d4)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#e65100")  # orange
))

fig.add_trace(go.Scatter(
    x=rto_pred_3["datetime_ending_ept"],
    y=rto_pred_3["MW_pred"],
    mode="lines",
    name=f"MW_pred ({fmt_mmdd(d3)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#ff5f00")  # orange
))

fig.add_trace(go.Scatter(
    x=rto_pred_2["datetime_ending_ept"],
    y=rto_pred_2["MW_pred"],
    mode="lines",
    name=f"MW_pred ({fmt_mmdd(d2)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#ff7f0e")  # orange
))

fig.add_trace(go.Scatter(
    x=rto_pred_1["datetime_ending_ept"],
    y=rto_pred_1["MW_pred"],
    mode="lines",
    name=f"MW_pred ({fmt_mmdd(d1)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#ff9f3a")  # yellow-orange
))

fig.add_trace(go.Scatter(
    x=rto_pred_0["datetime_ending_ept"],
    y=rto_pred_0["MW_pred"],
    mode="lines",
    name=f"MW_pred ({fmt_mmdd(d0)})",
    line=dict(width=2, shape ="spline", smoothing=sm_fct, color="#ffc078")  # green
))

fig.add_trace(go.Scatter(
    x=rto_zone_pred["datetime_ending_ept"],
    y=rto_zone_pred["MW_pred"],
    mode="lines",
    name=f"MW_Aggzone_pred ({fmt_mmdd(d0)})",
    line=dict(width=2, shape ="spline", smoothing=sm_fct, color="#ffd8a8")  # light green
))

# Line 3: MW_pred 
fig.add_trace(go.Scatter(
    x=rto_sector_4["datetime_ending_ept"],
    y=rto_sector_4["total_MW_pred"],
    mode="lines",
    name=f"MW_sct_pred ({fmt_mmdd(d4)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#0d3d66")  # darker blue
))

fig.add_trace(go.Scatter(
    x=rto_sector_3["datetime_ending_ept"],
    y=rto_sector_3["total_MW_pred"],
    mode="lines",
    name=f"MW_sct_pred ({fmt_mmdd(d3)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color = "#0b4f8a") # dark blue 
))

fig.add_trace(go.Scatter(
    x=rto_sector_2["datetime_ending_ept"],
    y=rto_sector_2["total_MW_pred"],
    mode="lines",
    name=f"MW_sct_pred ({fmt_mmdd(d2)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#1f77b4")  # blue
))

fig.add_trace(go.Scatter(
    x=rto_sector_1["datetime_ending_ept"],
    y=rto_sector_1["total_MW_pred"],
    mode="lines",
    name=f"MW_sct_pred ({fmt_mmdd(d1)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#4fa3d9")  # cyan
))

fig.add_trace(go.Scatter(
    x=rto_sector_0["datetime_ending_ept"],
    y=rto_sector_0["total_MW_pred"],
    mode="lines",
    name=f"MW_sct_pred ({fmt_mmdd(d0)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#9ecae1")  # blue
))

fig.add_trace(go.Scatter(
    x=rto_zone_sector["datetime_ending_ept"],
    y=rto_zone_sector["total_MW_pred"],
    mode="lines",
    name=f"MW_Aggzone_sct_pred ({fmt_mmdd(d0)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#b6d9eb")  # light blue
))

# Line 4: actual load
fig.add_trace(go.Scatter(
    x=actual_pjm["datetime_ending_ept"],
    y=actual_pjm["load_mw_hourly_avg"],
    mode="lines",
    name="Actual Load (PJM RTO)",
    line=dict(width=3, shape="spline", smoothing=sm_fct, color="#d62728")  # red
))

fig.update_layout(
    title="RTO Load Forecast vs Predictions vs Actual",
    xaxis_title="Datetime (EPT)",
    yaxis_title="MW",
    template="plotly_white",
    height=500,  # taller chart
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.add_annotation(
    text="Source: Chuhan Shi",
    xref="paper",
    yref="paper",
    x=0,
    y=-0.15,
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    font=dict(size=12, color="gray")
)

today_str = today.strftime("%m%d")
pio.write_html(fig, f"graph/rto_load/RTO_Load_5d_{today_str}.html")


################################ zone #####################################
zone_pred_4 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d4)}_to_{fmt_mmdd(d4 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d4)}.csv")
zone_pred_3 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d3)}_to_{fmt_mmdd(d3 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d3)}.csv")
zone_pred_2 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d2)}_to_{fmt_mmdd(d2 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d2)}.csv")
zone_pred_1 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d1)}_to_{fmt_mmdd(d1 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d1)}.csv")
zone_pred_0 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d0)}_to_{fmt_mmdd(d0 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d0)}.csv")

zone_pred_4["datetime_ending_ept"] = pd.to_datetime(zone_pred_4["datetime_ending_ept"])
zone_pred_3["datetime_ending_ept"] = pd.to_datetime(zone_pred_3["datetime_ending_ept"])
zone_pred_2["datetime_ending_ept"] = pd.to_datetime(zone_pred_2["datetime_ending_ept"])
zone_pred_1["datetime_ending_ept"] = pd.to_datetime(zone_pred_1["datetime_ending_ept"])
zone_pred_0["datetime_ending_ept"] = pd.to_datetime(zone_pred_0["datetime_ending_ept"])

zone_sector_4 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d4)}_to_{fmt_mmdd(d4 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d4)}.csv")
zone_sector_3 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d3)}_to_{fmt_mmdd(d3 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d3)}.csv")
zone_sector_2 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d2)}_to_{fmt_mmdd(d2 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d2)}.csv")
zone_sector_1 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d1)}_to_{fmt_mmdd(d1 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d1)}.csv")
zone_sector_0 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d0)}_to_{fmt_mmdd(d0 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d0)}.csv")

zone_sector_4["datetime_ending_ept"] = pd.to_datetime(zone_sector_4["datetime_ending_ept"])
zone_sector_3["datetime_ending_ept"] = pd.to_datetime(zone_sector_3["datetime_ending_ept"])
zone_sector_2["datetime_ending_ept"] = pd.to_datetime(zone_sector_2["datetime_ending_ept"])
zone_sector_1["datetime_ending_ept"] = pd.to_datetime(zone_sector_1["datetime_ending_ept"])
zone_sector_0["datetime_ending_ept"] = pd.to_datetime(zone_sector_0["datetime_ending_ept"])

areas = ["AE", "AEP", "APS", "ATSI", "BC", "COMED", "DAYTON", "DEOK", "DOM", "DPL", "DUQ", "EKPC", "JC", "ME", "PE", "PN", "PEP", "PL", "PS", "RECO"]
hourly_load_0 = hourly_load_0[hourly_load_0["area"].isin(areas)].copy()

zone_to_hourly = {
    "AE": "AE",
    "AEP": "AEP",
    "APS": "APS",
    "ATSI": "ATSI",
    "BGE": "BC",
    "COMED": "COMED",
    "DAYTON": "DAYTON",
    "DPL": "DPL",
    "DQE": "DUQ",
    "DUKE": "DEOK",
    "EKPC": "EKPC",
    "JCPL": "JC",
    "METED": "ME",
    "PECO": "PE",
    "PEPCO": "PEP",
    "PENLC": "PN",
    "PL": "PL",
    "PS": "PS",
    "RECO": "RECO",
    "VEPCO": "DOM"
}


zone_pred_0['area'] = zone_pred_0['zone'].map(zone_to_hourly)
zone_pred_1['area'] = zone_pred_1['zone'].map(zone_to_hourly)
zone_pred_2['area'] = zone_pred_2['zone'].map(zone_to_hourly)
zone_pred_3['area'] = zone_pred_3['zone'].map(zone_to_hourly)
zone_pred_4['area'] = zone_pred_4['zone'].map(zone_to_hourly)

zone_sector_0['area'] = zone_sector_0['zone'].map(zone_to_hourly)
zone_sector_1['area'] = zone_sector_1['zone'].map(zone_to_hourly)
zone_sector_2['area'] = zone_sector_2['zone'].map(zone_to_hourly)
zone_sector_3['area'] = zone_sector_3['zone'].map(zone_to_hourly)
zone_sector_4['area'] = zone_sector_4['zone'].map(zone_to_hourly)

############################### temperature line ############################
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

zonal_weather = zonal_weather[zonal_weather['zone'] != "UGI"].copy()

zonal_weather["area"] = zonal_weather["zone"].map(zone_to_hourly)


sm_fct = 0.5  # smoothing

today = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize()

cut_d_1_start = today - pd.Timedelta(days=3)                           # T-3 00:00:00
cut_d_1_end = today - pd.Timedelta(days=2) - pd.Timedelta(seconds=1)  # T-3 23:59:59
cut_d0_start = today - pd.Timedelta(days=2)                           # T-2 00:00:00
cut_d0_end = today - pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # T-2 23:59:59
cut_d1_start = today - pd.Timedelta(days=1)                           # T-1 00:00:00
cut_d1_end   = today - pd.Timedelta(seconds=1)                        # T-1 23:59:59
cut_d2_start = today                                                  # T 00:00:00
cut_d2_end   = today + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) # T 23:59:59
cut_d3_start = today + pd.Timedelta(days=1)                           # T+1 00:00:00
cut_d3_end   = today + pd.Timedelta(days=2) 


def calc_mae_rmse(df, pred_col, actual_col="load_mw_hourly_avg"):
    tmp = df[[pred_col, actual_col]].dropna().copy()
    if tmp.empty:
        return np.nan, np.nan
    err = tmp[pred_col] - tmp[actual_col]
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    return mae, rmse


all_areas = sorted(hourly_load_0["area"].dropna().unique())
areas_to_plot = all_areas[:20]   # force 20 x 1 layout

subplot_titles = []

# First pass: build subplot titles
for selected_area in areas_to_plot:
    actual_zone = (
        hourly_load_0[
            (hourly_load_0["area"] == selected_area) &
            (
                hourly_load_0["datetime_ending_ept"] >=
                ((pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() - pd.Timedelta(days=1))
            )
        ]
        .sort_values("datetime_ending_ept")
        .copy()
    )

    actual_d1 = actual_zone[
        (actual_zone["datetime_ending_ept"] >= cut_d1_start) &
        (actual_zone["datetime_ending_ept"] <= cut_d1_end)
    ][["datetime_ending_ept", "load_mw_hourly_avg"]].copy()

    zp2 = zone_pred_2[zone_pred_2["area"] == selected_area].copy()
    zs2 = zone_sector_2[zone_sector_2["area"] == selected_area].copy()

    forecast_part1 = zp2[
        (zp2["datetime_ending_ept"] >= cut_d1_start) &
        (zp2["datetime_ending_ept"] <= cut_d1_end)
    ][["datetime_ending_ept", "forecast_load_mw"]]

    zp2_d1 = zp2[
        (zp2["datetime_ending_ept"] >= cut_d1_start) &
        (zp2["datetime_ending_ept"] <= cut_d1_end)
    ][["datetime_ending_ept", "MW_pred"]].copy()

    zs2_d1 = zs2[
        (zs2["datetime_ending_ept"] >= cut_d1_start) &
        (zs2["datetime_ending_ept"] <= cut_d1_end)
    ][["datetime_ending_ept", "total_MW_pred"]].copy()

    pjm_d1 = forecast_part1.copy()

    eval_mw = actual_d1.merge(zp2_d1, on="datetime_ending_ept", how="left")
    eval_sct = actual_d1.merge(zs2_d1, on="datetime_ending_ept", how="left")
    eval_pjm = actual_d1.merge(pjm_d1, on="datetime_ending_ept", how="left")

    mae_mw, rmse_mw = calc_mae_rmse(eval_mw, "MW_pred")
    mae_sct, rmse_sct = calc_mae_rmse(eval_sct, "total_MW_pred")
    mae_pjm, rmse_pjm = calc_mae_rmse(eval_pjm, "forecast_load_mw")

    subtitle = (
        f"{selected_area} | "
        f"MW_pred MAE={mae_mw:.1f}, RMSE={rmse_mw:.1f} | "
        f"MW_sct_pred MAE={mae_sct:.1f}, RMSE={rmse_sct:.1f} | "
        f"PJM MAE={mae_pjm:.1f}, RMSE={rmse_pjm:.1f}"
    )
    subplot_titles.append(subtitle)

# Create 20 x 1 subplot figure
fig = make_subplots(
    rows=20,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.01,
    subplot_titles=subplot_titles,
    specs=[[{"secondary_y": True}] for _ in range(20)]
)

# Second pass: add traces
for i, selected_area in enumerate(areas_to_plot, start=1):
    # Filter actuals
    actual_zone = (
        hourly_load_0[
            (hourly_load_0["area"] == selected_area) &
            (
                hourly_load_0["datetime_ending_ept"] >=
                ((pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() - pd.Timedelta(days=3))
            )
        ]
        .sort_values("datetime_ending_ept")
        .copy()
    )

    # Filter prediction data
    zp4 = zone_pred_4[zone_pred_4["area"] == selected_area].copy()
    zp3 = zone_pred_3[zone_pred_3["area"] == selected_area].copy()
    zp2 = zone_pred_2[zone_pred_2["area"] == selected_area].copy()
    zp1 = zone_pred_1[zone_pred_1["area"] == selected_area].copy()
    zp0 = zone_pred_0[zone_pred_0["area"] == selected_area].copy()

    zs4 = zone_sector_4[zone_sector_4["area"] == selected_area].copy()
    zs3 = zone_sector_3[zone_sector_3["area"] == selected_area].copy()
    zs2 = zone_sector_2[zone_sector_2["area"] == selected_area].copy()
    zs1 = zone_sector_1[zone_sector_1["area"] == selected_area].copy()
    zs0 = zone_sector_0[zone_sector_0["area"] == selected_area].copy()

    # Build stitched PJM forecast
    forecast_part_1 = zp4[
        (zp4["datetime_ending_ept"] >= cut_d_1_start) &
        (zp4["datetime_ending_ept"] <= cut_d_1_end)
    ][["datetime_ending_ept", "forecast_load_mw"]]

    forecast_part0 = zp3[
        (zp3["datetime_ending_ept"] >= cut_d0_start) &
        (zp3["datetime_ending_ept"] <= cut_d0_end)
    ][["datetime_ending_ept", "forecast_load_mw"]]
 
    forecast_part1 = zp2[
        (zp2["datetime_ending_ept"] >= cut_d1_start) &
        (zp2["datetime_ending_ept"] <= cut_d1_end)
    ][["datetime_ending_ept", "forecast_load_mw"]]

    forecast_part2 = zp1[
        (zp1["datetime_ending_ept"] >= cut_d2_start) &
        (zp1["datetime_ending_ept"] <= cut_d2_end)
    ][["datetime_ending_ept", "forecast_load_mw"]]

    forecast_part3 = zp0[
        zp0["datetime_ending_ept"] >= cut_d3_start
    ][["datetime_ending_ept", "forecast_load_mw"]]

    forecast_combined = pd.concat(
        [forecast_part_1, forecast_part0, forecast_part1, forecast_part2, forecast_part3],
        ignore_index=True
    ).sort_values("datetime_ending_ept")

    # trim display series
    zp4 = zp4[zp4["datetime_ending_ept"] >= cut_d_1_start].copy()
    zs4 = zs4[zs4["datetime_ending_ept"] >= cut_d_1_start].copy()

    zw = zonal_weather[zonal_weather["area"] == selected_area].copy()

    # Optional: align time window with actual_zone
    zw = zw[
        (zw["datetime_ending_ept"] >= cut_d_1_start) &
        (zw["datetime_ending_ept"] <= cut_d3_end)
    ]

    # show legend only once
    show_legend = (i == 1)

    fig.add_trace(
        go.Scatter(
            x=forecast_combined["datetime_ending_ept"],
            y=forecast_combined["forecast_load_mw"],
            mode="lines",
            name="PJM Forecast",
            line=dict(width=2.5, shape="spline", smoothing=sm_fct, color="#000000"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zp4["datetime_ending_ept"],
            y=zp4["MW_pred"],
            mode="lines",
            name=f"MW_pred ({fmt_mmdd(d4)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#e65100"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zp3["datetime_ending_ept"],
            y=zp3["MW_pred"],
            mode="lines",
            name=f"MW_pred ({fmt_mmdd(d3)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#ff5f00"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zp2["datetime_ending_ept"],
            y=zp2["MW_pred"],
            mode="lines",
            name=f"MW_pred ({fmt_mmdd(d2)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#ff7f0e"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zp1["datetime_ending_ept"],
            y=zp1["MW_pred"],
            mode="lines",
            name=f"MW_pred ({fmt_mmdd(d1)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#ff9f3a"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zp0["datetime_ending_ept"],
            y=zp0["MW_pred"],
            mode="lines",
            name=f"MW_pred ({fmt_mmdd(d0)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#ffc078"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zs4["datetime_ending_ept"],
            y=zs4["total_MW_pred"],
            mode="lines",
            name=f"MW_sct_pred ({fmt_mmdd(d4)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#0d3d66"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zs3["datetime_ending_ept"],
            y=zs3["total_MW_pred"],
            mode="lines",
            name=f"MW_sct_pred ({fmt_mmdd(d3)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#0b4f8a"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zs2["datetime_ending_ept"],
            y=zs2["total_MW_pred"],
            mode="lines",
            name=f"MW_sct_pred ({fmt_mmdd(d2)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#1f77b4"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zs1["datetime_ending_ept"],
            y=zs1["total_MW_pred"],
            mode="lines",
            name=f"MW_sct_pred ({fmt_mmdd(d1)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#4fa3d9"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=zs0["datetime_ending_ept"],
            y=zs0["total_MW_pred"],
            mode="lines",
            name=f"MW_sct_pred ({fmt_mmdd(d0)})",
            line=dict(width=1.8, shape="spline", smoothing=sm_fct, color="#9ecae1"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=actual_zone["datetime_ending_ept"],
            y=actual_zone["load_mw_hourly_avg"],
            mode="lines",
            name="Actual Load",
            line=dict(width=2.5, shape="spline", smoothing=sm_fct, color="#d62728"),
            showlegend=show_legend
        ),
        row=i, col=1
    )

    fig.add_trace(
    go.Scatter(
        x=zw["datetime_ending_ept"],
        y=zw["temperature_2m"],
        mode="lines",
        name="Temperature (2m)",
        line=dict(width=1.5, dash="dot", color="#2ca02c"),
        showlegend=show_legend
    ),
    row=i, col=1,
    secondary_y=True   # <-- key
    )
    fig.update_xaxes(showticklabels=True)
    fig.update_yaxes(title_text="MW", row=i, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=i, col=1, secondary_y=True)

# Layout
fig.update_layout(
    title="Zone Load Forecast vs Predictions vs Actual",
    template="plotly_white",
    height=20 * 280,   # adjust as needed
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.005,
        xanchor="left",
        x=0
    ),
    margin=dict(t=120, l=60, r=30, b=40)
)

fig.add_annotation(
    text="Source: Chuhan Shi",
    xref="paper",
    yref="paper",
    x=0,
    y=-0.005,
    showarrow=False,
    xanchor="left",
    yanchor="top",
    font=dict(size=12, color="gray")
)

fig.update_xaxes(title_text="Datetime (EPT)", row=20, col=1)

pio.write_html(fig, f"graph/zone_load/Zone_Load_5d_{today_str}.html")

















