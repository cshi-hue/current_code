import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Targets
base_case_12_target = 85000
base_case_18_target = 87000


node_cty_map = pd.read_csv('data/pnode_county_mapping.csv')
node_cty_map.columns = (
    node_cty_map.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

dtype_map = {
    "agg_pjm_ExternalNodeID": "int64",
    "pjm_ExternalNodeID": "int64",
    "agg_NodeName": "string",
    "NodeName": "string",
    "forecast_area": "string",
    "is_valid": "int8"
}

offset = 0
today_str = (datetime.today() - timedelta(days=offset)).strftime("%m%d") 

hourly_load = pd.read_csv(
    f"data/hourly_load_MW_by_distribution/hourly_load_MW_by_distribution_{today_str}.csv",
    skiprows=[1],
    dtype=dtype_map,
    parse_dates=["datetime_ending_ept", "insert_datetime"]
)

hourly_load.sort_values("datetime_ending_ept", inplace=True)

hourly_load = hourly_load[hourly_load["datetime_ending_ept"] >= "2026-03-12"].copy()

hourly_load.columns = (
    hourly_load.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

hourly_load = hourly_load.drop(columns=[
    "autokey",
    "insert_datetime"
], errors="ignore")

hourly_load["distribution_factor"] = (
    hourly_load["distribution_factor"]
    .astype(float)
    .round(8)
)


hourly_load = hourly_load.dropna(subset=[
    "datetime_ending_ept",
    "nodename",
    "distributed_load_mw"
])

hourly_load = hourly_load.drop_duplicates(
    subset=["datetime_ending_ept", "nodename"]
)

hourly_load = hourly_load[["datetime_ending_ept", "agg_nodename", "nodename", "forecast_load_mw", "distributed_load_mw"]]

pnode_to_bus = pd.read_csv('data/XDAI_pnode_with_loc.csv')
pnode_to_bus.columns = (
    pnode_to_bus.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

hourly_load_bus_node = hourly_load.merge(pnode_to_bus[["psse_bus_#", "substation", "nodename"]], on = "nodename", how = "left")

bus_load = (
    hourly_load_bus_node
    .groupby(["datetime_ending_ept", "psse_bus_#", "substation"], as_index=False)
    ["distributed_load_mw"]
    .sum()
    .rename(columns={"distributed_load_mw": "bus_distributed_load_mw"})
)

hourly_load_bus = hourly_load_bus_node.merge(bus_load, on = ["datetime_ending_ept", "psse_bus_#", "substation"], how = "left")

hourly_load_zone_bus = hourly_load_bus[["datetime_ending_ept", "agg_nodename", "forecast_load_mw", "psse_bus_#", "substation", "bus_distributed_load_mw"]].drop_duplicates()

hourly_load_zone_bus["bus_distribution_factor"] = hourly_load_zone_bus["bus_distributed_load_mw"] / hourly_load_zone_bus["forecast_load_mw"]

load_forecast = pd.read_csv(f'data/load_frcstd_7_day/load_frcstd_7_day_{today_str}.csv', skiprows=[1])

load_forecast['forecast_load_mw'] = pd.to_numeric(load_forecast['forecast_load_mw'], errors='coerce')

load_forecast = load_forecast[['evaluated_at_datetime_ept', 'forecast_datetime_ending_ept', 'forecast_area', 'forecast_load_mw']].copy()

load_forecast["forecast_datetime_ending_ept"] = pd.to_datetime(load_forecast["forecast_datetime_ending_ept"])

forecast_to_agg = {
    "AE/MIDATL": "AECO",
    "AEP": "AEP",
    "AP": "APS",
    "ATSI": "ATSI",
    "BG&E/MIDATL": "BGE",
    "COMED": "COMED",
    "DAYTON": "DAY",
    "DEOK": "DEOK",
    "DOMINION": "DOM",
    "DP&L/MIDATL": "DPL",
    "DUQUESNE": "DUQ",
    "EKPC": "EKPC",
    "JCP&L/MIDATL": "JCPL",
    "METED/MIDATL": "METED",
    "PECO/MIDATL": "PECO",
    "PENELEC/MIDATL": "PENELEC",
    "PEPCO/MIDATL": "PEPCO",
    "PPL/MIDATL": "PPL",
    "PSE&G/MIDATL": "PSEG",
    "RECO/MIDATL": "RECO",
    "UGI/MIDATL": "UGI",
    
    # Regions
    "WESTERN_REGION": "WEST_REGION",
    "SOUTHERN_REGION": "SOUTH_REGION",
    "MID_ATLANTIC_REGION": "MIDATL_REGION",

    # Whole system
    "RTO_COMBINED": "RTO"
}

load_forecast["agg_nodename"] = load_forecast["forecast_area"].map(forecast_to_agg)

load_forecast = (
    load_forecast
    .sort_values(["forecast_datetime_ending_ept", "agg_nodename", "evaluated_at_datetime_ept"])
    .drop_duplicates(subset=["forecast_datetime_ending_ept", "agg_nodename"], keep="last")
    .reset_index(drop=True)
)

zone_load_forecast = load_forecast[load_forecast['agg_nodename'].isin(['AECO', 'AEP', 'APS', 'ATSI', 'BGE', 'COMED', 'DAY', 'DEOK', 'DOM', 'DPL', 'DUQ', 'EKPC', 'JCPL', 'METED', 'PENELEC', 'PEPCO', 'PPL', 'PSEG', 'RECO', 'UGI'])].copy()

ugi_mw = (
    zone_load_forecast.loc[zone_load_forecast["agg_nodename"] == "UGI", ["forecast_datetime_ending_ept", "forecast_load_mw"]]
    .rename(columns={"forecast_load_mw": "ugi_forecast_load_mw"})
)

zone_load_forecast = zone_load_forecast.merge(ugi_mw, on="forecast_datetime_ending_ept", how="left")

zone_load_forecast.loc[zone_load_forecast["agg_nodename"] == "PL", "forecast_load_mw"] = (
    zone_load_forecast.loc[zone_load_forecast["agg_nodename"] == "PL", "forecast_load_mw"] +
    zone_load_forecast.loc[zone_load_forecast["agg_nodename"] == "PL", "ugi_forecast_load_mw"].fillna(0)
)

zone_load_forecast = (
    zone_load_forecast.loc[zone_load_forecast["agg_nodename"] != "UGI"]
    .drop(columns="ugi_forecast_load_mw")
    .reset_index(drop=True)
)

zone_load_forecast = zone_load_forecast[["forecast_datetime_ending_ept", "agg_nodename", "forecast_load_mw"]].copy()

zone_load_forecast.rename(columns={"forecast_datetime_ending_ept": "datetime_ending_ept"}, inplace=True)


# forecast rows that need to be appended
last_bus_dt = hourly_load_zone_bus["datetime_ending_ept"].max()

future_zone = zone_load_forecast[
    zone_load_forecast["datetime_ending_ept"] > last_bus_dt
].copy()

# historical distribution factors
hist_factor = hourly_load_zone_bus[
    [
        "datetime_ending_ept",
        "agg_nodename",
        "psse_bus_#",
        "substation",
        "bus_distribution_factor"
    ]
].copy()

# create forward-looking target timestamps:
# a historical factor at t-1d, t-2d, ..., t-7d can be used for target time t
factor_candidates = []

for d in range(1, 8):
    tmp = hist_factor.copy()
    tmp["datetime_ending_ept"] = tmp["datetime_ending_ept"] + pd.Timedelta(days=d)
    factor_candidates.append(tmp)

factor_candidates = pd.concat(factor_candidates, ignore_index=True)

# median factor across past 7 same-hour observations
median_factor = (
    factor_candidates
    .groupby(
        ["datetime_ending_ept", "agg_nodename", "psse_bus_#", "substation"],
        as_index=False
    )["bus_distribution_factor"]
    .median()
)

# join median distribution factor onto forecast
append_df = future_zone.merge(
    median_factor,
    on=["datetime_ending_ept", "agg_nodename"],
    how="left"
)

# calculate bus-level distributed load
append_df["bus_distributed_load_mw"] = (
    append_df["forecast_load_mw"] * append_df["bus_distribution_factor"]
)

# align columns
append_df = append_df[
    [
        "datetime_ending_ept",
        "agg_nodename",
        "forecast_load_mw",
        "psse_bus_#",
        "substation",
        "bus_distributed_load_mw",
        "bus_distribution_factor"
    ]
]

# append to original table
hourly_load_zone_bus_updated = pd.concat(
    [hourly_load_zone_bus, append_df],
    ignore_index=True
)

bus_load_fcst = hourly_load_zone_bus_updated[["datetime_ending_ept", "psse_bus_#", "substation", "bus_distributed_load_mw"]].copy()

bus_load_fcst.rename(columns={"psse_bus_#": "busnum", "substation": "busname", "bus_distributed_load_mw": "loadmw"}, inplace=True)

base_case = pd.read_csv('data/PWD/base case.csv', skiprows = [0])
base_case.columns = (
    base_case.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
base_case["loadmw"] = 0.0

bus_PWD = pd.read_csv('data/PWD/bus PWD 2627 V2_3.csv', skiprows = [0])
bus_PWD.columns = (
    bus_PWD.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)
bus_PWD = bus_PWD[["label", "busnum", "busname", "areaname", "zonename"]].copy()
bus_PWD['loadid'] = 1
bus_PWD['loadstatus'] = 'Closed'
bus_PWD['loadmw'] = 0.0
bus_PWD['label'] = bus_PWD['busnum'].astype(str) + '-' + bus_PWD['loadid'].astype(str)


# target timestamps: tomorrow 12:00 and 18:00
tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

target_times = [
    tomorrow + pd.Timedelta(hours=12),
    tomorrow + pd.Timedelta(hours=18),
]

# make sure datetime is datetime type
bus_load_fcst["datetime_ending_ept"] = pd.to_datetime(bus_load_fcst["datetime_ending_ept"])

def build_load_case_for_time(target_time, base_case, bus_PWD, bus_load_fcst):
    # 1. get bus forecast load for this target hour
    load_at_time = (
        bus_load_fcst[
            bus_load_fcst["datetime_ending_ept"] == target_time
        ][["busnum", "busname", "loadmw"]]
        .copy()
    )

    load_at_time["busnum"] = load_at_time["busnum"].astype(int)

    # If duplicate busnum rows exist, aggregate them
    load_at_time = (
        load_at_time
        .groupby(["busnum", "busname"], as_index=False)["loadmw"]
        .sum()
    )

    # 2. start from base_case
    case = base_case.copy()
    case["busnum"] = case["busnum"].astype(int)

    # 3. find busnums missing from base_case but available in bus_PWD
    fcst_busnums = set(load_at_time["busnum"])
    base_busnums = set(case["busnum"])

    missing_busnums = fcst_busnums - base_busnums

    append_rows = bus_PWD[
        bus_PWD["busnum"].astype(int).isin(missing_busnums)
    ].copy()

    append_rows["busnum"] = append_rows["busnum"].astype(int)

    # Append missing buses from bus_PWD
    case = pd.concat([case, append_rows], ignore_index=True)

    # 4. count number of loadids per busnum
    loadid_count = (
        case
        .groupby("busnum")["loadid"]
        .transform("count")
    )

    case["_n_loadid"] = loadid_count

    # 5. merge forecast load
    case = case.merge(
        load_at_time[["busnum", "loadmw"]].rename(columns={"loadmw": "_fcst_loadmw"}),
        on="busnum",
        how="left"
    )

    # 6. fill loadmw:
    # if one loadid: full load
    # if multiple loadids: equal split
    mask = case["_fcst_loadmw"].notna()

    case.loc[mask, "loadmw"] = (
        case.loc[mask, "_fcst_loadmw"] / case.loc[mask, "_n_loadid"]
    )

    # 7. clean helper columns
    case = case.drop(columns=["_fcst_loadmw", "_n_loadid"])

    return case


base_case_12 = build_load_case_for_time(
    target_times[0],
    base_case,
    bus_PWD,
    bus_load_fcst
)

base_case_18 = build_load_case_for_time(
    target_times[1],
    base_case,
    bus_PWD,
    bus_load_fcst
)


# Compute current sums
base_case_12_sum = base_case_12["loadmw"].sum()
base_case_18_sum = base_case_18["loadmw"].sum()

# Avoid divide-by-zero
if base_case_12_sum != 0:
    base_case_12["loadmw"] = base_case_12["loadmw"] * (base_case_12_target / base_case_12_sum)

if base_case_18_sum != 0:
    base_case_18["loadmw"] = base_case_18["loadmw"] * (base_case_18_target / base_case_18_sum)

def write_aux_load(df, filename):
    cols = [
        "label", "busnum", "busname", "areaname",
        "zonename", "loadid", "loadstatus", "loadmw"
    ]
    out = df[cols].copy()

    # clean types / embedded quotes
    for c in ["label", "busname", "areaname", "zonename", "loadid", "loadstatus"]:
        out[c] = out[c].astype(str).str.replace('"', "", regex=False)

    out["busnum"] = out["busnum"].astype(int)
    out["loadmw"] = out["loadmw"].astype(float)

    # quote text fields the way AUX expects
    out["label_q"] = out["label"].map(lambda x: f'"{x}"')
    out["busname_q"] = out["busname"].map(lambda x: f'"{x}"')
    out["areaname_q"] = out["areaname"].map(lambda x: f'"{x}"')
    out["zonename_q"] = out["zonename"].map(lambda x: f'"{x}"')
    out["loadid_q"] = out["loadid"].map(lambda x: f'"{str(x).rjust(2)}"')
    out["loadstatus_q"] = out["loadstatus"].map(lambda x: f'"{x}"')

    # dynamic widths for nice alignment
    w_label = max(len("Label"), out["label_q"].map(len).max())
    w_busnum = max(len("BusNum"), out["busnum"].astype(str).map(len).max())
    w_busname = max(len("BusName"), out["busname_q"].map(len).max())
    w_areaname = max(len("AreaName"), out["areaname_q"].map(len).max())
    w_zonename = max(len("ZoneName"), out["zonename_q"].map(len).max())
    w_loadid = max(len("LoadID"), out["loadid_q"].map(len).max())
    w_loadstatus = max(len("LoadStatus"), out["loadstatus_q"].map(len).max())
    w_loadmw = max(len("LoadMW"), out["loadmw"].map(lambda x: f"{x:.2f}").map(len).max())

    with open(filename, "w") as f:
        f.write('Load (Label,BusNum,BusName,AreaName,ZoneName,ID,Status,MW)\n')
        f.write('{\n')

        for _, r in out.iterrows():
            line = (
                f"{r['label_q']:<{w_label}} "
                f"{r['busnum']:>{w_busnum}} "
                f"{r['busname_q']:<{w_busname}} "
                f"{r['areaname_q']:<{w_areaname}} "
                f"{r['zonename_q']:<{w_zonename}} "
                f"{r['loadid_q']:<{w_loadid}} "
                f"{r['loadstatus_q']:<{w_loadstatus}} "
                f"{r['loadmw']:>{w_loadmw}.2f}\n"
            )
            f.write(line)

        f.write('}\n')

run_date = datetime.today() + timedelta(days=1)
date_str = run_date.strftime("%Y%m%d")

write_aux_load(
    base_case_12,
    f"data/PWD/pjm_bus_df_demand_2627_v2-3_{date_str}_1200.aux"
)

write_aux_load(
    base_case_18,
    f"data/PWD/pjm_bus_df_demand_2627_v2-3_{date_str}_1800.aux"
)





























