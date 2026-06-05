import pandas as pd
import string

from pathlib import Path

from datetime import datetime, timedelta

dtype_map = {
    "agg_pjm_ExternalNodeID": "string",
    "pjm_ExternalNodeID": "string",
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

hourly_load = hourly_load[["datetime_ending_ept", "agg_nodename", "pjm_externalnodeid", "nodename", "distribution_factor", "forecast_area", "forecast_load_mw", "distributed_load_mw"]].copy()

# Step 1: Replace nodename
hourly_load.loc[
    hourly_load["nodename"] == "SHAWVILL34 KV   3TX_LD",
    "nodename"
] = "SHAWVILL34 KV   CLR2"

# Step 2: Aggregate
hourly_load = (
    hourly_load.groupby(["datetime_ending_ept", "nodename"], as_index=False)
      .agg({
          "agg_nodename": "first",
          "pjm_externalnodeid": "first",
          "distribution_factor": "sum",
          "forecast_area": "first",
          "forecast_load_mw": "first",
          "distributed_load_mw": "sum"
      })
)


hourly_load["clean_nodename"] = (
    hourly_load["nodename"]
    .str.replace(r"[ _-]", "", regex=True)
    .str.upper()
    .str.replace(r"(LD|LOAD)(\d*)$", r"\2", regex=True)  # remove suffix
)

dtype_map = {
    "psse_bus_#": "string",
    "substation": "string",
    "nodename": "string",
    "nodekey": "string",
    "externalnodeid": "string",
    "lon": "float64",
    "lat": "float64",
}

pnode_to_bus = pd.read_csv('data/pnode_with_loc.csv')
pnode_to_bus.columns = (
    pnode_to_bus.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

pnode_to_bus = pnode_to_bus.astype({
    "externalnodeid": "Int64",
    "nodekey": "Int64"  # optional, if numeric
})

pnode_to_bus = pnode_to_bus.astype(dtype_map)

############################################################
nodename_mapping = {
    "BENJAMIN138 KV  BENDATLD": "BENJAMIN138 KV  DATCENLD",
    "960 ELGI4 KV    LAUXBLUE": "960 ELGI4 KV    BLUAUXLD",
    "960 ELGI4 KV    LAUXRED": "960 ELGI4 KV    REDAUXLD",
    "13 CRAWF138 KV  ATR57R04": "13 CRAWF138 KV  TR57R04",
    "13 CRAWF138 KV  ATR58R04": "13 CRAWF138 KV  TR58R04",
    "WARDAV  138 KV  COLONIAL": "WARDAV  138 KV  COLONILD",
    "PANTHER 69 KV   CRYPTOLD": "PANTHER 69 KV   CRYPT_LD",
    "SCRUBGRS13 KV   DATACEN1": "SCRUBGRS13 KV   DATCENT1",
    ############## from DAI
    "135 ELMH138 KV  TR76_12": "135 ELMH138 KV  TR76_34",
    "SHAWVILL34 KV   14 TX": "SHAWVILL34 KV   CLR2",
    "ENGLISHT35 KV   BK 5": "ENGLISHT34.5 KV A209",
    "ENGLISHT35 KV   BK 2": "ENGLISHT34.5 KV D82",
    "ENGLISHT35 KV   BK 1": "ENGLISHT34.5 KV G111",
    "SMITHBUR35 KV   LOAD1": "SMITHBUR34.5 KV TR3",
    "WIND JC 35 KV   BANK1": "WIND JC 34.5 KV G215_LD",
    "WIND JC 35 KV   BANK3": "WIND JC 34.5 KV H60",
    "WYCKOFF 115 KV  BANK3": "WYCKOFF 34.5 KV D82_LD",
    "CAPITALA138 KV  T2": "CAPITALA12 KV   T2",
    "BROSVILL138 KV  CITYDANV": "BROSVILL69 KV   CITYDANV",
    "POKAGON 138 KV  T2": "POKAGON 138 KV  T4"
}

pnode_to_bus["nodename"] = pnode_to_bus["nodename"].replace(nodename_mapping)
#############################################################

pnode_to_bus["clean_nodename"] = (
    pnode_to_bus["nodename"]
    .str.replace(r"[ _-]", "", regex=True)
    .str.upper()
    .str.replace(r"(LD|LOAD)(\d*)$", r"\2", regex=True)  # remove suffix
)

cols = ["psse_bus_#", "substation", "nodename", "externalnodeid", "clean_nodename"]

pnode_map = pnode_to_bus[cols].copy()

hourly_load2 = hourly_load.copy()
hourly_load2["pjm_externalnodeid"] = hourly_load2["pjm_externalnodeid"].astype("Int64")
pnode_map["externalnodeid"] = pnode_map["externalnodeid"].astype("Int64")

# Optional but recommended: prevent duplicate name matches
pnode_map_by_name = pnode_map.drop_duplicates("clean_nodename", keep="first")

# 1) Match by externalnodeid
mapped = hourly_load2.merge(
    pnode_map,
    left_on="pjm_externalnodeid",
    right_on="externalnodeid",
    how="left",
    suffixes=("", "_bus")
)

miss_ext = mapped["psse_bus_#"].isna()

# 2) Match only missed rows by clean_nodename, preserving original row index
name_matches = (
    mapped.loc[miss_ext, ["clean_nodename"]]
    .reset_index(names="_row")
    .merge(
        pnode_map_by_name,
        on="clean_nodename",
        how="left",
        suffixes=("", "_bus")
    )
    .set_index("_row")
)

# Fill missed rows by index alignment
for col in cols:
    mapped.loc[name_matches.index, col] = (
        mapped.loc[name_matches.index, col]
        .combine_first(name_matches[col])
    )
    
hourly_load_mapped = mapped

mapped_hourly_load = hourly_load_mapped[
    hourly_load_mapped["psse_bus_#"].notna()
].copy()

hourly_load_bus_node = mapped_hourly_load[["datetime_ending_ept", "agg_nodename", "nodename", "distribution_factor", "forecast_area", "forecast_load_mw", "distributed_load_mw", "psse_bus_#", "substation"]].copy()

# drop unwanted column
df = hourly_load_bus_node.drop(columns=["forecast_area"]).copy()

df["psse_bus_#"] = df["psse_bus_#"].astype("int64")

# reorder columns
df = df[
    [
        "datetime_ending_ept",
        "psse_bus_#",
        "substation",
        "nodename",
        "distribution_factor",
        "distributed_load_mw",
        "agg_nodename",
        "forecast_load_mw"
    ]
]

# sort values
df = df.sort_values(
    by=[
        "datetime_ending_ept",   # primary
        "psse_bus_#",            # ascending
        "nodename"               # alphabetical
    ],
    ascending=[True, True, True]
).reset_index(drop=True)

# rename
df = df.rename(columns={"psse_bus_#": "BusNum"})

# IDs: X1-X9, then XA-XZ
id_pool = [f"X{i}" for i in range(1, 10)] + [f"X{c}" for c in string.ascii_uppercase]

# assign ID within each BusNum by nodename order
node_id_map = (
    df[["BusNum", "nodename"]]
    .drop_duplicates()
    .sort_values(["BusNum", "nodename"])
    .assign(
        ID=lambda x: x.groupby("BusNum").cumcount().map(lambda i: id_pool[i])
    )
)

create_aux = df.merge(node_id_map, on=["BusNum", "nodename"], how="left")

# required AUX columns
create_aux["LabelAppend"] = create_aux["BusNum"].astype(str) + "-" + create_aux["ID"]
create_aux["LoadSMvar"] = float("0.0")
create_aux["LoadSMW"] = float("0.0")
create_aux["Status"] = "Closed"

create_aux = create_aux[["LabelAppend", "BusNum", "ID", "LoadSMvar", "LoadSMW", "Status"]].drop_duplicates()

base_case = pd.read_csv('data/PWD/base case.csv', skiprows = [0])
base_case = base_case[["Label", "BusNum", "AreaName", "LoadID", "LoadMW",  "LoadStatus"]]
base_case.rename(columns={"Label": "LabelAppend", "LoadID": "ID", "LoadMW": "LoadSMW", "LoadStatus": "Status"}, inplace=True)
base_case["ID"] = base_case["ID"].astype(str)
base_case = base_case[base_case["AreaName"] != "PJM"].copy()

# Find BusNums already in create_aux
existing_busnums = set(create_aux["BusNum"])

# Filter base_case for rows NOT in create_aux
missing_rows = base_case[~base_case["BusNum"].isin(existing_busnums)].copy()

# Align columns (important)
missing_rows = missing_rows.reindex(columns=create_aux.columns)

# Append
create_aux_updated = pd.concat([create_aux, missing_rows], ignore_index=True)

# build lines (fast)
lines = (
    '"' 
    + create_aux_updated["LabelAppend"].fillna("").astype(str)
    + '" '
    + create_aux_updated["BusNum"].fillna("").astype(str)
    + ' "'
    + create_aux_updated["ID"].fillna("").astype(str)
    + '" '
    + create_aux_updated["LoadSMvar"].fillna(0).astype(str)
    + " "
    + create_aux_updated["LoadSMW"].fillna(0).astype(str)
    + ' "'
    + create_aux_updated["Status"].fillna("").astype(str)
    + '"'
).astype(str).tolist()

# assemble full AUX text
aux_text = (
    "Load (LabelAppend,BusNum,ID,SMvar,SMW,Status)\n"
    "{\n"
    + "\n".join(lines)
    + "\n}"
)

date_str = pd.Timestamp.today().normalize().strftime("%Y%m%d")


output_dirs = [
    Path(r"data/PWD/load_create"),
    Path(r"\\GC-NAS\QuantTeam\cshi\PWD\load_create"),
]

# write AUX files
for out_dir in output_dirs:
    out_dir.mkdir(parents=True, exist_ok=True)

    aux_path = out_dir / f"load_create_{date_str}.aux"

    with open(aux_path, "w", encoding="utf-8") as f:
        f.write(aux_text)


create_aux_updated = create_aux_updated.rename(columns={
    "LabelAppend": "LabelAppend",
    "BusNum": "BusNum",
    "ID": "ID",
    "LoadSMvar": "SMvar",
    "LoadSMW": "SMW",
    "Status": "Status"
})

create_aux_updated["LabelAppend"] = ""

create_aux_updated = create_aux_updated.dropna(how="all")


# write CSV files
for out_dir in output_dirs:
    csv_path = out_dir / f"load_create_{date_str}.csv"

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Load\n")

        create_aux_updated.to_csv(
            f,
            index=False,
            lineterminator="\n"
        )






































































