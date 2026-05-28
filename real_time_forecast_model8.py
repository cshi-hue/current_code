import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Targets
he12_target = 85000
he18_target = 87000

# data import
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

hourly_load = hourly_load[["datetime_ending_ept", "nodename", "distributed_load_mw"]]


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


bus_load["datetime_ending_ept"] = pd.to_datetime(bus_load["datetime_ending_ept"])

bus_load["hour"] = bus_load["datetime_ending_ept"].dt.hour
bus_load["date"] = bus_load["datetime_ending_ept"].dt.date

# Keep only HE12 and HE18
bus_load = bus_load[bus_load["hour"].isin([12, 18])].copy()

# Latest 7 existing dates
latest_7_dates = sorted(bus_load["date"].dropna().unique())[-7:]

hist = bus_load[bus_load["date"].isin(latest_7_dates)].copy()

# Prediction function
def predict_next_day(group):
    """
    group columns expected:
      - date
      - bus_distributed_load_mw
    """
    g = (
        group[["date", "bus_distributed_load_mw"]]
        .dropna()
        .sort_values("date", ascending=False)
        .drop_duplicates(subset=["date"], keep="first")
        .copy()
    )

    if g.empty:
        return 0.0

    # Closest to farthest after trimming
    g = g.sort_values("date", ascending=False).copy()

    base_weights = np.array([0.35, 0.15, 0.10, 0.05, 0.05, 0.10, 0.30], dtype=float)
    weights = base_weights[:len(g)]

    # weights = weights / weights.sum()

    pred = float((g["bus_distributed_load_mw"].to_numpy() * weights).sum())
    return pred

# Predict per substation, per HE
pred_df = (
    hist.groupby(["psse_bus_#", "substation", "hour"], as_index=False)
        .apply(lambda x: pd.Series({"pred_loadmw": predict_next_day(x)}))
        .reset_index(drop=True)
)

# Split HE12 / HE18 predictions into dicts for easy mapping
pred12_map = (
    pred_df[pred_df["hour"] == 12]
    .set_index("psse_bus_#")["pred_loadmw"]
    .to_dict()
)

pred18_map = (
    pred_df[pred_df["hour"] == 18]
    .set_index("psse_bus_#")["pred_loadmw"]
    .to_dict()
)

# If one row per loadid already, counting rows is enough.
# If needed, use nunique on loadid for safety.
loadid_count = (
    base_case.groupby("busnum")["loadid"]
    .nunique()
    .to_dict()
)

def build_output(base_case_df, pred_map, bus_PWD):
    # Step 1: Start with base
    out = base_case_df.copy()

    # Step 2: Identify missing busnums from prediction
    pred_busnums = set(pred_map.keys())
    existing_busnums = set(out["busnum"].unique())

    missing_busnums = pred_busnums - existing_busnums

    # Step 3: Pull missing rows from bus_PWD
    if missing_busnums:
        add_rows = bus_PWD[bus_PWD["busnum"].isin(missing_busnums)].copy()

        # Optional: warn if some busnums still missing
        found_busnums = set(add_rows["busnum"].unique())
        still_missing = missing_busnums - found_busnums
        if still_missing:
            print(f"Warning: {len(still_missing)} busnums not found in bus_PWD")

        # Append
        out = pd.concat([out, add_rows], ignore_index=True)

    # Step 4: Map predictions
    out["bus_total_pred"] = out["busnum"].map(pred_map).fillna(0.0)

    # Step 5: Recompute loadid count AFTER augmentation
    loadid_count = out.groupby("busnum")["loadid"].transform("count")

    # Step 6: Allocate
    out["loadmw"] = out["bus_total_pred"] / loadid_count

    # Step 7: Cleanup
    out.drop(columns=["bus_total_pred"], inplace=True)

    return out

he12_df = build_output(base_case, pred12_map, bus_PWD)
he18_df = build_output(base_case, pred18_map, bus_PWD)

# Compute current sums
he12_sum = he12_df["loadmw"].sum()
he18_sum = he18_df["loadmw"].sum()

# Avoid divide-by-zero
if he12_sum != 0:
    he12_df["loadmw"] = he12_df["loadmw"] * (he12_target / he12_sum)

if he18_sum != 0:
    he18_df["loadmw"] = he18_df["loadmw"] * (he18_target / he18_sum)

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
    he12_df,
    f"data/PWD/pjm_bus_demand_2627_v2-3_{date_str}_1200.aux"
)

write_aux_load(
    he18_df,
    f"data/PWD/pjm_bus_demand_2627_v2-3_{date_str}_1800.aux"
)

print(f"Saved to data/PWD")































