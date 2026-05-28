import numpy as np
import pandas as pd
import string
import pyodbc

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

hourly_load.loc[
    hourly_load["nodename"] == "SHAWVILL34 KV   3TX_LD",
    "nodename"
] = "SHAWVILL34 KV   CLR2"

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

hourly_load = hourly_load.rename(columns={
    "pjm_externalnodeid": "externalnodeid"
})

hourly_load["clean_nodename"] = (
    hourly_load["nodename"]
    .str.replace(r"[ _]", "", regex=True)
    .str.upper()
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
hourly_load2["externalnodeid"] = hourly_load2["externalnodeid"].astype("Int64")
pnode_map["externalnodeid"] = pnode_map["externalnodeid"].astype("Int64")

# Optional but recommended: prevent duplicate name matches
pnode_map_by_name = pnode_map.drop_duplicates("clean_nodename", keep="first")

# 1) Match by externalnodeid
mapped = hourly_load2.merge(
    pnode_map,
    on="externalnodeid",
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

zone_load_forecast = load_forecast[load_forecast['agg_nodename'].isin(['AECO', 'AEP', 'APS', 'ATSI', 'BGE', 'COMED', 'DAY', 'DEOK', 'DOM', 'DPL', 'DUQ', 'EKPC', 'JCPL', 'METED', 'PECO', 'PENELEC', 'PEPCO', 'PPL', 'PSEG', 'RECO', 'UGI'])].copy()

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
last_dt = create_aux["datetime_ending_ept"].max()

future_zone = zone_load_forecast[
    zone_load_forecast["datetime_ending_ept"] > last_dt
].copy()

# historical distribution factors
hist_factor = create_aux[
    [
        "datetime_ending_ept",
        "BusNum",
        "substation",
        "nodename",
        "distribution_factor",
        "agg_nodename",
        "ID"
    ]
].copy()

# create forward-looking target timestamps:
# a historical factor at t-1d, t-2d, ..., t-7d can be used for target time t
factor_candidates = []

for d in range(1, 8):
    tmp = hist_factor.copy()
    tmp["datetime_ending_ept"] = tmp["datetime_ending_ept"] + pd.Timedelta(days=d)
    tmp["lag_day"] = d
    factor_candidates.append(tmp)

    # add one extra copy for yesterday and same day last week
    if d in [1, 7, 14]:
        tmp_extra = hist_factor.copy()
        tmp_extra["datetime_ending_ept"] = tmp_extra["datetime_ending_ept"] + pd.Timedelta(days=d)
        tmp_extra["lag_day"] = d
        factor_candidates.append(tmp_extra)

factor_candidates = pd.concat(factor_candidates, ignore_index=True)

# median factor across past 7 same-hour observations
median_factor = (
    factor_candidates
    .groupby(
        ["datetime_ending_ept", "BusNum", "substation", "nodename", "agg_nodename", "ID"],
        as_index=False
    )["distribution_factor"]
    .median()
)

# join median distribution factor onto forecast
append_df = future_zone.merge(
    median_factor,
    on=["datetime_ending_ept", "agg_nodename"],
    how="left"
)

# calculate bus-level distributed load
append_df["distributed_load_mw"] = (
    append_df["forecast_load_mw"] * append_df["distribution_factor"]
)

# align columns
append_df = append_df[
    [
        "datetime_ending_ept",
        "BusNum",
        "substation",
        "nodename",
        "distribution_factor",
        "distributed_load_mw",
        "agg_nodename",
        "forecast_load_mw",
        "ID"
    ]
]

# append to original table
hourly_load_forecast = pd.concat(
    [create_aux, append_df],
    ignore_index=True
)

hourly_load_forecast['distributed_load_mw'] = hourly_load_forecast['distributed_load_mw'].round(5)

hourly_load_forecast["BusNum"] = pd.to_numeric(
    hourly_load_forecast["BusNum"],
    errors="coerce"
)

hourly_load_forecast.sort_values(
    by=["datetime_ending_ept", "BusNum"],
    ascending=[True, True],
    inplace=True
)

today_str = datetime.today().strftime("%Y%m%d")

create_aux_updated = pd.read_csv(
    f"data/PWD/load_create/load_create_{today_str}.csv", skiprows=1
)

# define tomorrow
tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
date_str = tomorrow.strftime("%Y%m%d")

target_12 = tomorrow + pd.Timedelta(hours=12)
target_18 = tomorrow + pd.Timedelta(hours=18)

def make_load_aux(df, out_path):
    create_aux = (
        df[["BusNum", "ID", "distributed_load_mw"]]
        .drop_duplicates()
        .copy()
    )

    load_aux = create_aux_updated.merge(
        create_aux[['BusNum', 'ID', 'distributed_load_mw']],
        on=['BusNum', 'ID'],
        how='left'
    )

    # Update SMW where a match exists
    load_aux['SMW'] = load_aux['distributed_load_mw'].combine_first(
        load_aux['SMW']
    )

    # Optional: remove helper column
    load_aux.drop(columns=['distributed_load_mw'], inplace=True)

    load_aux = load_aux[["BusNum", "ID", "SMW"]].copy()

    lines = (
        load_aux["BusNum"].astype(int).astype(str)
        + ' "'
        + load_aux["ID"].astype(str)
        + '" '
        + " "
        + load_aux["SMW"].astype(str)
    )

    aux_text = (
        "Load (BusNum,ID,SMW)\n"
        "{\n"
        + "\n".join(lines)
        + "\n}"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(aux_text)

    return load_aux

# # Targets
# target_sum_12 = 77000
# target_sum_18 = 80000


# 12:00 rows
df_12 = hourly_load_forecast[
    hourly_load_forecast["datetime_ending_ept"] == target_12
].copy()

# current_sum_12 = df_12["distributed_load_mw"].sum()
# scale_12 = target_sum_12 / current_sum_12

# print(current_sum_12, scale_12)

# df_12["distributed_load_mw"] = df_12["distributed_load_mw"] * scale_12

# 18:00 rows
df_18 = hourly_load_forecast[
    hourly_load_forecast["datetime_ending_ept"] == target_18
].copy()

# current_sum_18 = df_18["distributed_load_mw"].sum()
# scale_18 = target_sum_18 / current_sum_18

# print(current_sum_18, scale_18)

# df_18["distributed_load_mw"] = df_18["distributed_load_mw"] * scale_18

# dynamic paths
path_12 = f"data/PWD/load_HE/load_{date_str}_HE12.aux"
path_18 = f"data/PWD/load_HE/load_{date_str}_HE18.aux"

# create files
create_aux_12 = make_load_aux(df_12, path_12)
create_aux_18 = make_load_aux(df_18, path_18)

def make_load_csv(create_aux, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("Load\n")
        create_aux.to_csv(
            f,
            index=False,
            lineterminator="\n"
        )


csv_path_12 = f"data/PWD/load_HE/load_{date_str}_HE12.csv"
csv_path_18 = f"data/PWD/load_HE/load_{date_str}_HE18.csv"

# create CSV files
make_load_csv(create_aux_12, csv_path_12)
make_load_csv(create_aux_18, csv_path_18)

######################################################## upload to SQL  ########################################################


offset = 0
today_str = (datetime.today() - timedelta(days=offset)).strftime("%m%d") 

hourly_load = pd.read_csv(
    f"data/hourly_load_MW_by_distribution/hourly_load_MW_by_distribution_{today_str}.csv",
    skiprows=[1],
    dtype=dtype_map,
    parse_dates=["datetime_ending_ept", "insert_datetime"]
)

# Ensure datetime type
hourly_load_forecast["datetime_ending_ept"] = pd.to_datetime(
    hourly_load_forecast["datetime_ending_ept"]
)

hourly_load["datetime_ending_ept"] = pd.to_datetime(
    hourly_load["datetime_ending_ept"]
)

# Tomorrow filter
today = pd.Timestamp.today().normalize()
tomorrow = today + pd.Timedelta(days=1)
day_after = tomorrow + pd.Timedelta(days=1)

forecast_df = hourly_load_forecast[
    (hourly_load_forecast["datetime_ending_ept"] >= tomorrow) &
    (hourly_load_forecast["datetime_ending_ept"] < day_after)
].copy()

# Build mapping from hourly_load using NodeName
# (drop duplicates just in case)
node_mapping = (
    hourly_load[
        [
            "NodeName",
            "agg_pjm_ExternalNodeID",
            "pjm_ExternalNodeID"
        ]
    ]
    .dropna(subset=["NodeName"])
    .drop_duplicates(subset=["NodeName"])
)

# Merge mapping into forecast data
forecast_df = forecast_df.merge(
    node_mapping,
    left_on="nodename",
    right_on="NodeName",
    how="left"
)

# Build final dataframe in hourly_load schema
hourly_load_fcst = pd.DataFrame({
    "autokey": "",
    "datetime_ending_ept": forecast_df["datetime_ending_ept"],
    "agg_pjm_ExternalNodeID": forecast_df["agg_pjm_ExternalNodeID"],
    "agg_NodeName": forecast_df["agg_nodename"],
    "pjm_ExternalNodeID": forecast_df["pjm_ExternalNodeID"],
    "NodeName": forecast_df["nodename"],
    "distribution_factor": forecast_df["distribution_factor"],
    "forecast_area": forecast_df["agg_nodename"],
    "forecast_load_mw": forecast_df["forecast_load_mw"],
    "distributed_load_mw": forecast_df["distributed_load_mw"],
    "is_valid": 1,
    "insert_datetime": ""
})

# Optional: enforce same column order
hourly_load_fcst = hourly_load_fcst[hourly_load.columns]

# drop columns
hourly_load_fcst = hourly_load_fcst.drop(columns=["autokey", "insert_datetime"])

tmr_str = tomorrow.strftime("%y%m%d")

hourly_load_fcst.to_csv(f"data/hourly_load_MW_by_distribution_factors_frcst/hourly_load_MW_by_distribution_factors_frcst_{tmr_str}.csv", index=False)


conn = pyodbc.connect(
    "DRIVER={ODBC Driver 13 for SQL Server};"
    "SERVER=10.1.10.243;"
    "DATABASE=QA;"
    "UID=cshi;"
    "PWD=Dkh$2910;"
    "TrustServerCertificate=yes;"
)

cursor = conn.cursor()
cursor.fast_executemany = True

upload_df = hourly_load_fcst.copy()

upload_df = upload_df[
    [
        "datetime_ending_ept",
        "agg_pjm_ExternalNodeID",
        "agg_NodeName",
        "pjm_ExternalNodeID",
        "NodeName",
        "distribution_factor",
        "forecast_area",
        "forecast_load_mw",
        "distributed_load_mw",
        "is_valid",
    ]
]

upload_df["insert_datetime"] = pd.Timestamp.now()

# Get current max autokey from SQL table
cursor.execute("""
    SELECT ISNULL(MAX([autokey]), 0)
    FROM [QA].[dbo].[hourly_load_MW_by_distribution_factors_frcst]
""")

max_autokey = cursor.fetchone()[0]

# Add autokey as first column
upload_df.insert(
    0,
    "autokey",
    range(max_autokey + 1, max_autokey + 1 + len(upload_df))
)

# Convert pandas/numpy values to native Python values
def to_python_value(x):
    if pd.isna(x):
        return None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    return x

data = [
    tuple(to_python_value(x) for x in row)
    for row in upload_df.itertuples(index=False, name=None)
]

sql = """
INSERT INTO [QA].[dbo].[hourly_load_MW_by_distribution_factors_frcst] (
    [autokey],
    [datetime_ending_ept],
    [agg_pjm_ExternalNodeID],
    [agg_NodeName],
    [pjm_ExternalNodeID],
    [NodeName],
    [distribution_factor],
    [forecast_area],
    [forecast_load_mw],
    [distributed_load_mw],
    [is_valid],
    [insert_datetime]
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

try:
    cursor.executemany(sql, data)
    conn.commit()
    print(f"Uploaded {len(upload_df):,} rows.")

except pyodbc.Error as e:
    conn.rollback()
    print("Upload failed:")
    print(e)

finally:
    cursor.close()
    conn.close()






































