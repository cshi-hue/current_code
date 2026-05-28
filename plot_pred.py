import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from datetime import datetime, timedelta

########################################## RTO plot ##########################################
offset = 0
today = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize()

d0 = today
d1 = today - pd.Timedelta(days=1)
d2 = today - pd.Timedelta(days=2)

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

rto_pred_0["datetime_ending_ept"] = pd.to_datetime(rto_pred_0["datetime_ending_ept"])
rto_pred_1["datetime_ending_ept"] = pd.to_datetime(rto_pred_1["datetime_ending_ept"])
rto_pred_2["datetime_ending_ept"] = pd.to_datetime(rto_pred_2["datetime_ending_ept"])


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

rto_sector_0["datetime_ending_ept"] = pd.to_datetime(rto_sector_0["datetime_ending_ept"])
rto_sector_1["datetime_ending_ept"] = pd.to_datetime(rto_sector_1["datetime_ending_ept"])
rto_sector_2["datetime_ending_ept"] = pd.to_datetime(rto_sector_2["datetime_ending_ept"])

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
        (hourly_load_0["datetime_ending_ept"] >= ((pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize() - pd.Timedelta(days=1)))
    ]
    .sort_values("datetime_ending_ept")
    .copy()
)

today = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize()

cut_d1_start = today - pd.Timedelta(days=1)                   # T-1 00:00:00    

cut_d1_end = today - pd.Timedelta(seconds=1)                  # T-1 23:59:59
cut_d2_start = today                                          # T 00:00:00

cut_d2_end = today + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)  # T 23:59:59
cut_d3_start = today + pd.Timedelta(days=1)                   # T+1 00:00:00

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

forecast_combined = pd.concat([forecast_part1, forecast_part2, forecast_part3])

rto_pred_2 = rto_pred_2[rto_pred_2["datetime_ending_ept"] >= cut_d1_start].copy()
rto_sector_2 = rto_sector_2[rto_sector_2["datetime_ending_ept"] >= cut_d1_start].copy()


sm_fct = 0.5  # Adjust this value for more or less smoothing

# helper functions
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def first_day_metrics(pred_df, pred_col, actual_df, actual_col="load_mw_hourly_avg"):
    # Inner join on timestamp so only overlapping hours are compared
    df = pd.merge(
        pred_df[["datetime_ending_ept", pred_col]].copy(),
        actual_df[["datetime_ending_ept", actual_col]].copy(),
        on="datetime_ending_ept",
        how="inner"
    ).sort_values("datetime_ending_ept")

    # First day = first calendar day present in the merged comparison set
    first_day = df["datetime_ending_ept"].dt.floor("D").min()
    df_day1 = df[df["datetime_ending_ept"].dt.floor("D") == first_day].copy()

    return {
        "date": first_day,
        "mae": mae(df_day1[actual_col], df_day1[pred_col]),
        "rmse": rmse(df_day1[actual_col], df_day1[pred_col]),
        "n": len(df_day1),
    }

# calculate metrics for day 1 
m_pred2 = first_day_metrics(rto_pred_2, "MW_pred", actual_pjm)
m_sector2 = first_day_metrics(rto_sector_2, "total_MW_pred", actual_pjm)
m_forecast = first_day_metrics(forecast_combined, "forecast_load_mw", actual_pjm)

metrics_text = (
    f"<b>Error Metrics</b><br>"
    f"Date: {m_pred2['date']:%Y-%m-%d}<br>"
    f"MW_pred ({fmt_mmdd(d2)}): MAE={m_pred2['mae']:.1f}, RMSE={m_pred2['rmse']:.1f}<br>"
    f"MW_sct_pred ({fmt_mmdd(d2)}): MAE={m_sector2['mae']:.1f}, RMSE={m_sector2['rmse']:.1f}<br>"
    f"PJM Forecast: MAE={m_forecast['mae']:.1f}, RMSE={m_forecast['rmse']:.1f}"
)

fig = go.Figure()

# Line 1: stitched forecast
fig.add_trace(go.Scatter(
    x=forecast_combined["datetime_ending_ept"],
    y=forecast_combined["forecast_load_mw"],
    mode="lines",
    name="PJM Forecast",
    line=dict(width=3, shape="spline", smoothing=sm_fct, color="#000000")
))

# Line 2: MW_pred from d2
fig.add_trace(go.Scatter(
    x=rto_pred_2["datetime_ending_ept"],
    y=rto_pred_2["MW_pred"],
    mode="lines",
    name=f"MW_pred ({fmt_mmdd(d2)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#ff7f0e")
))

# Line 3: MW_pred from d1
fig.add_trace(go.Scatter(
    x=rto_pred_1["datetime_ending_ept"],
    y=rto_pred_1["MW_pred"],
    mode="lines",
    name=f"MW_pred ({fmt_mmdd(d1)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#ff9f3a")
))

fig.add_trace(go.Scatter(
    x=rto_pred_0["datetime_ending_ept"],
    y=rto_pred_0["MW_pred"],
    mode="lines",
    name=f"MW_pred ({fmt_mmdd(d0)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#ffc078")
))

# Sector preds
fig.add_trace(go.Scatter(
    x=rto_sector_2["datetime_ending_ept"],
    y=rto_sector_2["total_MW_pred"],
    mode="lines",
    name=f"MW_sct_pred ({fmt_mmdd(d2)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#1f77b4")
))

fig.add_trace(go.Scatter(
    x=rto_sector_1["datetime_ending_ept"],
    y=rto_sector_1["total_MW_pred"],
    mode="lines",
    name=f"MW_sct_pred ({fmt_mmdd(d1)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#4fa3d9")
))

fig.add_trace(go.Scatter(
    x=rto_sector_0["datetime_ending_ept"],
    y=rto_sector_0["total_MW_pred"],
    mode="lines",
    name=f"MW_sct_pred ({fmt_mmdd(d0)})",
    line=dict(width=2, shape="spline", smoothing=sm_fct, color="#9ecae1")
))

# Actual load
fig.add_trace(go.Scatter(
    x=actual_pjm["datetime_ending_ept"],
    y=actual_pjm["load_mw_hourly_avg"],
    mode="lines",
    name="Actual Load (PJM RTO)",
    line=dict(width=3, shape="spline", smoothing=sm_fct, color="#d62728")
))

fig.update_layout(
    title="RTO Load Forecast vs Predictions vs Actual",
    xaxis_title="Datetime (EPT)",
    yaxis_title="MW",
    template="plotly_white",
    height=700,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    annotations=[
        dict(
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=12),
            text=metrics_text
        )
    ]
)

today_str = today.strftime("%m%d")
pio.write_html(fig, f"graph/RTO_Load_{today_str}.html")

################################# zone plot ##########################################

zone_pred_2 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d2)}_to_{fmt_mmdd(d2 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d2)}.csv")
zone_pred_1 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d1)}_to_{fmt_mmdd(d1 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d1)}.csv")
zone_pred_0 = pd.read_csv(f"data/prediction/zone_prediction_{fmt_mmdd(d0)}_to_{fmt_mmdd(d0 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d0)}.csv")

zone_pred_2["datetime_ending_ept"] = pd.to_datetime(zone_pred_2["datetime_ending_ept"])
zone_pred_1["datetime_ending_ept"] = pd.to_datetime(zone_pred_1["datetime_ending_ept"])
zone_pred_0["datetime_ending_ept"] = pd.to_datetime(zone_pred_0["datetime_ending_ept"])

zone_sector_2 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d2)}_to_{fmt_mmdd(d2 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d2)}.csv")
zone_sector_1 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d1)}_to_{fmt_mmdd(d1 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d1)}.csv")
zone_sector_0 = pd.read_csv(f"data/prediction/zone_sector_prediction_{fmt_mmdd(d0)}_to_{fmt_mmdd(d0 + pd.Timedelta(days=1))}_at_{fmt_yymmdd(d0)}.csv")


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

zone_sector_0['area'] = zone_sector_0['zone'].map(zone_to_hourly)
zone_sector_1['area'] = zone_sector_1['zone'].map(zone_to_hourly)
zone_sector_2['area'] = zone_sector_2['zone'].map(zone_to_hourly)


sm_fct = 0.5  # smoothing

today = (pd.Timestamp.today() - pd.Timedelta(days=offset)).normalize()

cut_d1_start = today - pd.Timedelta(days=1)                           # T-1 00:00:00
cut_d1_end   = today - pd.Timedelta(seconds=1)                        # T-1 23:59:59
cut_d2_start = today                                                  # T 00:00:00
cut_d2_end   = today + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) # T 23:59:59
cut_d3_start = today + pd.Timedelta(days=1)                           # T+1 00:00:00


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


fig = make_subplots(
    rows=20,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.01,
    subplot_titles=subplot_titles
)

for i, selected_area in enumerate(areas_to_plot, start=1):
    # Filter actuals
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

    # Filter prediction data
    zp2 = zone_pred_2[zone_pred_2["area"] == selected_area].copy()
    zp1 = zone_pred_1[zone_pred_1["area"] == selected_area].copy()
    zp0 = zone_pred_0[zone_pred_0["area"] == selected_area].copy()

    zs2 = zone_sector_2[zone_sector_2["area"] == selected_area].copy()
    zs1 = zone_sector_1[zone_sector_1["area"] == selected_area].copy()
    zs0 = zone_sector_0[zone_sector_0["area"] == selected_area].copy()

    # Build stitched PJM forecast
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
        [forecast_part1, forecast_part2, forecast_part3],
        ignore_index=True
    ).sort_values("datetime_ending_ept")

    # trim display series
    zp2 = zp2[zp2["datetime_ending_ept"] >= cut_d1_start].copy()
    zs2 = zs2[zs2["datetime_ending_ept"] >= cut_d1_start].copy()

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

    fig.update_yaxes(title_text="MW", row=i, col=1)

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

fig.update_xaxes(title_text="Datetime (EPT)", row=20, col=1)

pio.write_html(fig, f"graph/Zone_Load_{today_str}.html")








