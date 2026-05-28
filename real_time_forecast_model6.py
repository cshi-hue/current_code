import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import geopandas as gpd
import json
from plotly.utils import PlotlyJSONEncoder

node_cty_map = pd.read_csv('data/pnode_county_mapping.csv')
node_cty_map.columns = (
    node_cty_map.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

offset = 0
today_str = (datetime.today() - timedelta(days=offset)).strftime("%m%d") 

dtype_map = {
    "agg_pjm_ExternalNodeID": "int64",
    "pjm_ExternalNodeID": "int64",
    "agg_NodeName": "string",
    "NodeName": "string",
    "forecast_area": "string",
    "is_valid": "int8"
}

hourly_load = pd.read_csv(
    f"data/hourly_load_MW_by_distribution/hourly_load_MW_by_distribution_{today_str}.csv",
    skiprows=[1],
    dtype=dtype_map,
    parse_dates=["datetime_ending_ept", "insert_datetime"]
)

hourly_load = hourly_load.sort_values("datetime_ending_ept").reset_index(drop=True)

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

hourly_load_county = hourly_load[["datetime_ending_ept", "agg_nodename", "nodename", "distribution_factor", "forecast_area", "forecast_load_mw", "distributed_load_mw"]].merge(node_cty_map[["nodename", "county_name", "latitude", "longitude", "state_short_name"]], on="nodename", how="inner")

county_load = (
    hourly_load_county
    .groupby(
        ["datetime_ending_ept", "county_name", "state_short_name"],
        as_index=False
    )
    .agg(
        distributed_load_mw=("distributed_load_mw", "sum"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean")
    )
    .sort_values(["datetime_ending_ept", "county_name"])
)

county_geo = (
    county_load
    .groupby("county_name", as_index=False)
    .agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean")
    )
)

# round to 2 decimals
county_geo["latitude"] = county_geo["latitude"].round(2)
county_geo["longitude"] = county_geo["longitude"].round(2)

county_load = county_load[["datetime_ending_ept", "county_name", "state_short_name", "distributed_load_mw"]].merge(county_geo, on="county_name", how="left").copy()

county_load = county_load[county_load["county_name"] != "IA_Scott"].copy()

county_load_weather = county_load.copy()

code_to_label = {
    'S': 'Solar',
    'W': 'Wind',
    'R': 'Res',
    'C': 'Com/Ind',
    'U': 'Unknown'
}

### PJM PART
county_to_class_name = {

    # DC
    'DC_District of Columbia': 'C',

    # DE
    'DE_Kent': 'R', 'DE_New Castle': 'R', 'DE_Sussex': 'R',

    # IA
    # 'IA_Scott': 'U',

    # IL
    'IL_Boone': 'R', 'IL_Cook': 'C', 'IL_DeKalb': 'R', 'IL_DuPage': 'C',
    'IL_Grundy': 'R', 'IL_Kane': 'C', 'IL_Kankakee': 'U', 'IL_Kendall': 'R',
    'IL_LaSalle': 'S', 'IL_Lake': 'C', 'IL_Lee': 'R', 'IL_Livingston': 'S',
    'IL_McHenry': 'R', 'IL_Ogle': 'R', 'IL_Stephenson': 'S', 'IL_Tazewell': 'U',
    'IL_Whiteside': 'S', 'IL_Will': 'C', 'IL_Winnebago': 'R',

    # IN
    'IN_Adams': 'U', 'IN_Allen': 'C', 'IN_Blackford': 'U', 'IN_DeKalb': 'U',
    'IN_Delaware': 'C', 'IN_Elkhart': 'C', 'IN_Franklin': 'U', 'IN_Grant': 'R',
    'IN_Henry': 'R', 'IN_Huntington': 'U', 'IN_Jay': 'R', 'IN_Kosciusko': 'U',
    'IN_LaPorte': 'C', 'IN_Madison': 'R', 'IN_Marshall': 'R', 'IN_Noble': 'R',
    'IN_Randolph': 'R', 'IN_Spencer': 'U', 'IN_St. Joseph': 'R', 'IN_Wabash': 'U',
    'IN_Wayne': 'S', 'IN_Wells': 'R', 'IN_Whitley': 'C',

    # KY
    'KY_Adair': 'U', 'KY_Anderson': 'R', 'KY_Barren': 'R', 'KY_Bath': 'R',
    'KY_Bell': 'R', 'KY_Boone': 'C', 'KY_Bourbon': 'R', 'KY_Boyd': 'R',
    'KY_Boyle': 'R', 'KY_Bracken': 'R', 'KY_Breathitt': 'R', 'KY_Bullitt': 'R',
    'KY_Campbell': 'R', 'KY_Carroll': 'U', 'KY_Carter': 'R', 'KY_Casey': 'R',
    'KY_Clark': 'S', 'KY_Clay': 'R', 'KY_Clinton': 'R', 'KY_Daviess': 'U',
    'KY_Elliott': 'U', 'KY_Estill': 'R', 'KY_Fayette': 'R', 'KY_Fleming': 'R',
    'KY_Floyd': 'U', 'KY_Franklin': 'R', 'KY_Gallatin': 'R', 'KY_Garrard': 'R',
    'KY_Grant': 'R', 'KY_Green': 'R', 'KY_Greenup': 'C', 'KY_Hardin': 'R',
    'KY_Harlan': 'C', 'KY_Harrison': 'R', 'KY_Hart': 'R', 'KY_Henry': 'R',
    'KY_Jackson': 'R', 'KY_Jefferson': 'C', 'KY_Jessamine': 'R', 'KY_Johnson': 'C',
    'KY_Kenton': 'C', 'KY_Knott': 'R', 'KY_Knox': 'R', 'KY_Larue': 'R',
    'KY_Laurel': 'R', 'KY_Lawrence': 'R', 'KY_Lee': 'R', 'KY_Leslie': 'R',
    'KY_Lewis': 'R', 'KY_Lincoln': 'R', 'KY_Madison': 'R', 'KY_Magoffin': 'R',
    'KY_Marion': 'R', 'KY_Martin': 'U', 'KY_Mason': 'U', 'KY_McCreary': 'U',
    'KY_Menifee': 'R', 'KY_Mercer': 'R', 'KY_Metcalfe': 'R', 'KY_Montgomery': 'R',
    'KY_Morgan': 'R', 'KY_Nelson': 'R', 'KY_Nicholas': 'R', 'KY_Owen': 'R',
    'KY_Owsley': 'U', 'KY_Pendleton': 'R', 'KY_Perry': 'R', 'KY_Pike': 'R',
    'KY_Powell': 'R', 'KY_Pulaski': 'R', 'KY_Rockcastle': 'R', 'KY_Rowan': 'U',
    'KY_Russell': 'R', 'KY_Scott': 'R', 'KY_Shelby': 'R', 'KY_Spencer': 'R',
    'KY_Taylor': 'U', 'KY_Trimble': 'R', 'KY_Washington': 'R', 'KY_Wayne': 'R',
    'KY_Whitley': 'C', 'KY_Wolfe': 'U', 'KY_Woodford': 'U',

    # MD
    'MD_Allegany': 'C', 'MD_Anne Arundel': 'R', 'MD_Baltimore': 'R', 'MD_Caroline': 'R',
    'MD_Carroll': 'R', 'MD_Cecil': 'R', 'MD_Charles': 'R', 'MD_Dorchester': 'R',
    'MD_Frederick': 'R', 'MD_Garrett': 'R', 'MD_Harford': 'R', 'MD_Howard': 'R',
    'MD_Kent': 'R', 'MD_Montgomery': 'C', 'MD_Prince George\'s': 'R', 'MD_Queen Anne\'s': 'R',
    'MD_Somerset': 'R', 'MD_Talbot': 'R', 'MD_Washington': 'R', 'MD_Wicomico': 'R',
    'MD_Worcester': 'R',

    # MI
    'MI_Berrien': 'C', 'MI_Cass': 'R', 'MI_Kalamazoo': 'U', 'MI_St. Joseph': 'U',
    'MI_Van Buren': 'R',

    # NC
    'NC_Beaufort': 'S', 'NC_Bertie': 'S', 'NC_Chowan': 'U', 'NC_Currituck': 'S',
    'NC_Dare': 'R', 'NC_Edgecombe': 'S', 'NC_Gates': 'U', 'NC_Halifax': 'S',
    'NC_Hertford': 'S', 'NC_Martin': 'S', 'NC_Nash': 'U', 'NC_Northampton': 'S',
    'NC_Pasquotank': 'U', 'NC_Perquimans': 'S', 'NC_Tyrrell': 'S', 'NC_Warren': 'S',
    'NC_Washington': 'U',

    # NJ
    'NJ_Atlantic': 'S', 'NJ_Bergen': 'R', 'NJ_Burlington': 'S', 'NJ_Camden': 'R',
    'NJ_Cape May': 'S', 'NJ_Cumberland': 'S', 'NJ_Essex': 'R', 'NJ_Gloucester': 'S',
    'NJ_Hudson': 'S', 'NJ_Hunterdon': 'S', 'NJ_Mercer': 'S', 'NJ_Middlesex': 'S',
    'NJ_Monmouth': 'S', 'NJ_Morris': 'R', 'NJ_Ocean': 'S', 'NJ_Passaic': 'R',
    'NJ_Salem': 'S', 'NJ_Somerset': 'S', 'NJ_Sussex': 'R', 'NJ_Union': 'R',
    'NJ_Warren': 'R',

    # NY
    'NY_New York': 'C', 'NY_Richmond': 'C', 'NY_Rockland': 'R',

    # OH
    'OH_Adams': 'C', 'OH_Allen': 'C', 'OH_Ashland': 'R', 'OH_Ashtabula': 'R',
    'OH_Athens': 'U', 'OH_Auglaize': 'R', 'OH_Belmont': 'R', 'OH_Brown': 'R',
    'OH_Butler': 'C', 'OH_Carroll': 'R', 'OH_Champaign': 'R', 'OH_Clark': 'R',
    'OH_Clermont': 'R', 'OH_Clinton': 'R', 'OH_Columbiana': 'R', 'OH_Coshocton': 'U',
    'OH_Crawford': 'R', 'OH_Cuyahoga': 'C', 'OH_Darke': 'R', 'OH_Defiance': 'U',
    'OH_Delaware': 'R', 'OH_Erie': 'C', 'OH_Fairfield': 'R', 'OH_Fayette': 'U',
    'OH_Franklin': 'C', 'OH_Fulton': 'W', 'OH_Gallia': 'R', 'OH_Geauga': 'U',
    'OH_Greene': 'C', 'OH_Guernsey': 'U', 'OH_Hamilton': 'C', 'OH_Hancock': 'C',
    'OH_Hardin': 'R', 'OH_Harrison': 'R', 'OH_Henry': 'W', 'OH_Highland': 'U',
    'OH_Hocking': 'U', 'OH_Holmes': 'C', 'OH_Huron': 'R', 'OH_Jackson': 'C',
    'OH_Jefferson': 'U', 'OH_Knox': 'R', 'OH_Lake': 'C', 'OH_Lawrence': 'C',
    'OH_Licking': 'C', 'OH_Logan': 'C', 'OH_Lorain': 'C', 'OH_Lucas': 'C',
    'OH_Madison': 'R', 'OH_Mahoning': 'C', 'OH_Marion': 'U', 'OH_Medina': 'C',
    'OH_Meigs': 'U', 'OH_Mercer': 'U', 'OH_Miami': 'R', 'OH_Monroe': 'U',
    'OH_Montgomery': 'C', 'OH_Morgan': 'U', 'OH_Morrow': 'R', 'OH_Muskingum': 'C',
    'OH_Noble': 'R', 'OH_Ottawa': 'U', 'OH_Paulding': 'W', 'OH_Perry': 'R',
    'OH_Pickaway': 'C', 'OH_Pike': 'C', 'OH_Portage': 'C', 'OH_Preble': 'U',
    'OH_Putnam': 'U', 'OH_Richland': 'C', 'OH_Ross': 'C', 'OH_Sandusky': 'C',
    'OH_Scioto': 'U', 'OH_Seneca': 'C', 'OH_Shelby': 'U', 'OH_Stark': 'C',
    'OH_Summit': 'C', 'OH_Trumbull': 'C', 'OH_Tuscarawas': 'U', 'OH_Union': 'C',
    'OH_Van Wert': 'R', 'OH_Vinton': 'R', 'OH_Warren': 'C', 'OH_Washington': 'C',
    'OH_Wayne': 'U', 'OH_Williams': 'U', 'OH_Wood': 'R', 'OH_Wyandot': 'R',

    # PA
    'PA_Adams': 'R', 'PA_Allegheny': 'C', 'PA_Armstrong': 'U', 'PA_Beaver': 'C',
    'PA_Bedford': 'R', 'PA_Berks': 'R', 'PA_Blair': 'U', 'PA_Bradford': 'R',
    'PA_Bucks': 'R', 'PA_Butler': 'U', 'PA_Cambria': 'R', 'PA_Carbon': 'U',
    'PA_Centre': 'C', 'PA_Clarion': 'C', 'PA_Clearfield': 'R', 'PA_Clinton': 'R',
    'PA_Columbia': 'R', 'PA_Crawford': 'R', 'PA_Cumberland': 'R', 'PA_Dauphin': 'R',
    'PA_Delaware': 'R', 'PA_Elk': 'U', 'PA_Erie': 'C', 'PA_Fayette': 'C',
    'PA_Franklin': 'R', 'PA_Fulton': 'C', 'PA_Greene': 'R', 'PA_Huntingdon': 'R',
    'PA_Indiana': 'R', 'PA_Jefferson': 'R', 'PA_Juniata': 'R', 'PA_Lackawanna': 'R',
    'PA_Lancaster': 'R', 'PA_Lawrence': 'R', 'PA_Lebanon': 'R', 'PA_Lehigh': 'R',
    'PA_Luzerne': 'R', 'PA_Lycoming': 'R', 'PA_McKean': 'C', 'PA_Mercer': 'R',
    'PA_Mifflin': 'U', 'PA_Monroe': 'R', 'PA_Montgomery': 'C', 'PA_Montour': 'R',
    'PA_Northampton': 'R', 'PA_Northumberland': 'R', 'PA_Perry': 'R', 'PA_Pike': 'R',
    'PA_Potter': 'U', 'PA_Schuylkill': 'R', 'PA_Snyder': 'R', 'PA_Somerset': 'U',
    'PA_Susquehanna': 'R', 'PA_Tioga': 'R', 'PA_Union': 'R', 'PA_Venango': 'R',
    'PA_Warren': 'U', 'PA_Washington': 'U', 'PA_Westmoreland': 'C', 'PA_Wyoming': 'U',
    'PA_York': 'R',

    # TN
    'TN_Hawkins': 'U', 'TN_Sullivan': 'R', 'TN_Washington': 'R',

    # VA
    'VA_Accomack': 'R', 'VA_Albemarle': 'R', 'VA_Alexandria': 'C', 'VA_Alleghany': 'U',
    'VA_Amelia': 'U', 'VA_Amherst': 'R', 'VA_Appomattox': 'R', 'VA_Arlington': 'C',
    'VA_Augusta': 'R', 'VA_Bedford': 'R', 'VA_Bland': 'U', 'VA_Botetourt': 'U',
    'VA_Brunswick': 'C', 'VA_Buchanan': 'U', 'VA_Buckingham': 'S', 'VA_Campbell': 'C',
    'VA_Caroline': 'R', 'VA_Carroll': 'R', 'VA_Charles City': 'C', 'VA_Charlotte': 'S',
    'VA_Chesapeake': 'R', 'VA_Chesterfield': 'R', 'VA_Clarke': 'U', 'VA_Covington': 'U',
    'VA_Craig': 'R', 'VA_Culpeper': 'C', 'VA_Cumberland': 'R', 'VA_Dickenson': 'R',
    'VA_Dinwiddie': 'U', 'VA_Emporia': 'C', 'VA_Essex': 'S', 'VA_Fairfax': 'C',
    'VA_Falls Church': 'C', 'VA_Fauquier': 'R', 'VA_Floyd': 'R', 'VA_Fluvanna': 'S',
    'VA_Franklin': 'C', 'VA_Frederick': '', 'VA_Fredericksburg': 'C', 'VA_Giles': 'R',
    'VA_Gloucester': 'C', 'VA_Goochland': 'R', 'VA_Grayson': 'R', 'VA_Greensville': 'U',
    'VA_Halifax': 'R', 'VA_Hampton': 'C', 'VA_Hanover': 'R', 'VA_Harrisonburg': 'C',
    'VA_Henrico': 'C', 'VA_Henry': 'S', 'VA_Hopewell': 'C', 'VA_Isle of Wight': 'R',
    'VA_James City': 'R', 'VA_King George': 'U', 'VA_King William': 'U', 'VA_Lancaster': 'S',
    'VA_Loudoun': 'C', 'VA_Louisa': 'S', 'VA_Lunenburg': 'R', 'VA_Lynchburg': 'C',
    'VA_Madison': 'U', 'VA_Manassas': 'C', 'VA_Manassas Park': 'C', 'VA_Martinsville': 'C',
    'VA_Mecklenburg': 'C', 'VA_Middlesex': 'R', 'VA_Montgomery': 'C', 'VA_Nelson': 'U',
    'VA_New Kent': 'R', 'VA_Newport News': 'C', 'VA_Norfolk': 'C', 'VA_Northampton': 'R',
    'VA_Nottoway': 'C', 'VA_Orange': 'R', 'VA_Page': 'R', 'VA_Patrick': 'U',
    'VA_Petersburg': 'R', 'VA_Pittsylvania': 'R', 'VA_Poquoson': 'C', 'VA_Portsmouth': 'C',
    'VA_Powhatan': 'S', 'VA_Prince Edward': 'S', 'VA_Prince George': 'R', 'VA_Prince William': 'C',
    'VA_Pulaski': 'C', 'VA_Rappahannock': 'U', 'VA_Richmond': 'C', 'VA_Roanoke': 'U',
    'VA_Rockbridge': 'R', 'VA_Rockingham': 'R', 'VA_Russell': 'R', 'VA_Salem': 'C',
    'VA_Scott': 'R', 'VA_Shenandoah': 'R', 'VA_Smyth': 'R', 'VA_Southampton': 'U',
    'VA_Spotsylvania': 'R', 'VA_Stafford': 'R', 'VA_Suffolk': 'R', 'VA_Surry': 'C',
    'VA_Sussex': 'C', 'VA_Tazewell': 'R', 'VA_Virginia Beach': 'R', 'VA_Warren': 'R',
    'VA_Washington': 'U', 'VA_Waynesboro': 'C', 'VA_Westmoreland': 'R', 'VA_Winchester': 'C',
    'VA_Wythe': 'S', 'VA_York': 'C',

    # WV
    'WV_Barbour': 'U', 'WV_Berkeley': 'R', 'WV_Boone': 'C', 'WV_Braxton': 'U',
    'WV_Brooke': 'C', 'WV_Cabell': 'C', 'WV_Calhoun': 'U', 'WV_Doddridge': 'U',
    'WV_Fayette': 'U', 'WV_Gilmer': 'R', 'WV_Grant': 'R', 'WV_Greenbrier': 'R',
    'WV_Hampshire': 'U', 'WV_Hancock': 'C', 'WV_Hardy': 'R', 'WV_Harrison': 'C',
    'WV_Jefferson': 'R', 'WV_Kanawha': 'C', 'WV_Lewis': 'U', 'WV_Lincoln': 'C',
    'WV_Logan': 'C', 'WV_Marion': 'R', 'WV_Marshall': 'C', 'WV_Mason': 'R',
    'WV_McDowell': 'C', 'WV_Mercer': 'C', 'WV_Mineral': 'R', 'WV_Mingo': 'C',
    'WV_Monongalia': 'R', 'WV_Morgan': 'R', 'WV_Nicholas': 'R', 'WV_Ohio': 'C',
    'WV_Pendleton': 'R', 'WV_Pleasants': 'U', 'WV_Pocahontas': 'R', 'WV_Preston': 'R',
    'WV_Putnam': 'R', 'WV_Raleigh': 'R', 'WV_Randolph': 'R', 'WV_Ritchie': 'U',
    'WV_Roane': 'R', 'WV_Summers': 'R', 'WV_Taylor': 'U', 'WV_Tucker': 'R',
    'WV_Tyler': 'U', 'WV_Upshur': 'R', 'WV_Wayne': 'U', 'WV_Webster': 'R',
    'WV_Wetzel': 'C', 'WV_Wood': 'C', 'WV_Wyoming': 'C',
}

county_load_weather['class_code'] = county_load_weather['county_name'].map(county_to_class_name).fillna('U')
county_load_weather['class_name'] = county_load_weather['class_code'].map(code_to_label)


# =========================================================
# 1. Load and prepare data
# =========================================================
df = county_load_weather.copy()
df["datetime"] = pd.to_datetime(df["datetime_ending_ept"])
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour

shapefile_path = "data/cb_2023_us_county_500k/cb_2023_us_county_500k.shp"
gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

# =========================================================
# 2. Yesterday peak load
# =========================================================
latest_date_overall = pd.to_datetime(df["date"]).max().date()
yesterday_date = latest_date_overall - pd.Timedelta(days=2)

yesterday_data = df[df["date"] == yesterday_date].copy()
yesterday_peaks = (
    yesterday_data.groupby("county_name")["distributed_load_mw"]
    .max()
    .reset_index()
    .rename(columns={"distributed_load_mw": "yesterday_peak_mw"})
)

# =========================================================
# 3. Prepare county-to-geometry join
# =========================================================
df_map = df.drop_duplicates(subset=["county_name"]).copy()
df_map = df_map.merge(yesterday_peaks, on="county_name", how="left")

df_map["county_clean"] = df_map["county_name"].apply(
    lambda x: x.split("_", 1)[-1] if "_" in x else x
)

state_fips_map = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56"
}

df_map["STATEFP"] = df_map["state_short_name"].map(state_fips_map)

df_joined = gdf.merge(
    df_map,
    left_on=["STATEFP", "NAME"],
    right_on=["STATEFP", "county_clean"],
    how="inner"
).copy()

# =========================================================
# 4. Build county time-series dictionary
# =========================================================
weights = {
    0: 0.55,   # today
    1: 0.20,   # yesterday
    6: 0.25,   # 6 days ago
    2: 0.10,
    3: -0.05,
    4: -0.05,
    5: 0.05
}

county_series = {}

for county in sorted(df["county_name"].dropna().unique()):
    df_county = df[df["county_name"] == county].copy()
    if df_county.empty:
        continue

    latest_date = pd.to_datetime(df_county["date"]).max().date() - pd.Timedelta(days=1)
    past_days = [latest_date - pd.Timedelta(days=i) for i in range(7)]
    df_county_7d = df_county[df_county["date"].isin(past_days)].copy()

    pred = pd.DataFrame({"hour": range(24)})
    pred["predicted_load"] = 0.0

    for lag, w in weights.items():
        d = latest_date - pd.Timedelta(days=lag)
        temp = df_county[df_county["date"] == d][["hour", "distributed_load_mw"]].copy()
        temp = temp.rename(columns={"distributed_load_mw": "load"})
        pred = pred.merge(temp, on="hour", how="left")
        pred["predicted_load"] += w * pred["load"].fillna(0)
        pred = pred.drop(columns=["load"])

    pred_date = latest_date + pd.Timedelta(days=1)

    traces = []

    # Define base color (blue here, can change)
    # base_rgb = (0, 114, 178)

    class_color_map = {
        "Com/Ind": (0, 114, 178),
        "Solar": (230, 159, 0),
        "Wind": (0, 158, 115),
        "Res": (204, 121, 167),
        "Unknown": (150, 150, 150)
    }

    base_rgb = class_color_map.get(df_county["class_name"].iloc[0], (0, 114, 178))

    dates_sorted = sorted(df_county_7d["date"].unique())

    for i, d in enumerate(dates_sorted):
        sub = df_county_7d[df_county_7d["date"] == d].sort_values("hour")

        # i = 0 → oldest, i = 6 → most recent
        # stronger color for recent days
        # alpha = 0.2 + 0.8 * (i / (len(dates_sorted) - 1))  # range ~0.2 → 1.0

        if len(dates_sorted) > 1:
            alpha = 0.2 + 0.8 * (i / (len(dates_sorted) - 1))
        else:
            alpha = 1.0

        color = f"rgba({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]}, {alpha})"

        traces.append({
            "x": sub["hour"].tolist(),
            "y": sub["distributed_load_mw"].tolist(),
            "mode": "lines",
            "name": str(d),
            "type": "scatter",
            "line": {
                "color": color,
                "width": 2
            }
        })

    traces.append({
        "x": pred["hour"].tolist(),
        "y": pred["predicted_load"].tolist(),
        "mode": "lines",
        "name": f"Prediction ({pred_date})",
        "type": "scatter",
        "line": {"dash": "dash", "width": 3, "color": "black"}
    })

    county_series[county] = traces

# =========================================================
# 5. Build county lookup for dropdown/map interaction
# =========================================================
county_lookup = {}

for _, row in df_joined.iterrows():
    geom = row.geometry
    point = geom.representative_point()  # safer than centroid for irregular polygons

    county_lookup[row["county_name"]] = {
        "geoid": row["GEOID"],
        "lat": float(point.y),
        "lon": float(point.x)
    }

county_options_html = "\n".join([
    f'<option value="{county}">{county}</option>'
    for county in sorted(county_lookup.keys())
])

# =========================================================
# 6. Build map figure
# =========================================================
fig_map = px.choropleth(
    df_joined,
    geojson=df_joined.__geo_interface__,
    locations="GEOID",
    featureidkey="properties.GEOID",
    color="class_name",
    color_discrete_map={
        "Com/Ind": "#0072B2",
        "Solar": "#E69F00",
        "Wind": "#009E73",
        "Res": "#CC79A7",
        "Unknown": "#999999"
    },
    custom_data=["county_name", "class_name", "yesterday_peak_mw"],
    labels={
        "county_name": "County",
        "class_name": "Class",
        "yesterday_peak_mw": "Yesterday Peak Load (MW)"
    }
)

fig_map.update_traces(
    hovertemplate=(
        "<b>%{customdata[0]}</b><br>"
        "Class: %{customdata[1]}<br>"
        "Yesterday Peak Load (MW): %{customdata[2]:.2f}<extra></extra>"
    )
)

fig_map.update_geos(fitbounds="locations", visible=False)
fig_map.update_layout(
    title="County Classification Map",
    margin={"r": 0, "t": 40, "l": 0, "b": 40},
    height=700
)

fig_map.add_annotation(
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

# =========================================================
# 7. Initial empty line chart
# =========================================================
fig_line = go.Figure()
fig_line.update_layout(
    title="Click a county on the map or choose from dropdown",
    xaxis_title="Hour",
    yaxis_title="Load (MW)",
    height=700,
    xaxis=dict(tickmode="linear", dtick=4),
    margin=dict(l=50, r=20, t=60, b=50)
)

# =========================================================
# 8. Export both figures into one interactive HTML
# =========================================================
map_json = json.dumps(fig_map.to_dict(), cls=PlotlyJSONEncoder)
line_json = json.dumps(fig_line.to_dict(), cls=PlotlyJSONEncoder)
series_json = json.dumps(county_series, cls=PlotlyJSONEncoder)
lookup_json = json.dumps(county_lookup, cls=PlotlyJSONEncoder)

html_str = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>County Load Map and Prediction</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
        }}
        .wrapper {{
            display: flex;
            flex-direction: row;
            width: 100%;
            height: 100vh;
        }}
        .left-panel {{
            width: 58%;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        .control-box {{
            padding: 10px 14px;
            border-bottom: 1px solid #ddd;
            background: #fafafa;
        }}
        .control-box label {{
            font-weight: bold;
            margin-right: 8px;
        }}
        .control-box select {{
            width: 320px;
            padding: 6px 8px;
            font-size: 14px;
        }}
        #map {{
            width: 100%;
            flex: 1;
        }}
        .map-note {{
            padding: 8px 14px 12px 14px;
            font-size: 13px;
            color: #555;
            background: #fff;
            border-top: 1px solid #eee;
            line-height: 1.4;
        }}

        #line {{
            width: 42%;
            height: 100vh;
        }}
    </style>
</head>
<body>
    <div class="wrapper">
        <div class="left-panel">
            <div class="control-box">
                <label for="countyDropdown">Select county:</label>
                <select id="countyDropdown">
                    <option value="">-- Choose a county --</option>
                    {county_options_html}
                </select>
            </div>
            <div id="map"></div>
            <div class="map-note">
                Note: Click a county on the map or choose one from the dropdown to view its load profile and prediction.
                </div>
        </div>
        <div id="line"></div>
    </div>

    <script>
        const mapFig = {map_json};
        const lineFig = {line_json};
        const countySeries = {series_json};
        const countyLookup = {lookup_json};

        Plotly.newPlot("map", mapFig.data, mapFig.layout, {{responsive: true}});
        Plotly.newPlot("line", lineFig.data, lineFig.layout, {{responsive: true}});

        const mapDiv = document.getElementById("map");
        const lineDiv = document.getElementById("line");
        const countyDropdown = document.getElementById("countyDropdown");

        const baseMapData = JSON.parse(JSON.stringify(mapFig.data));
        const baseMapLayout = JSON.parse(JSON.stringify(mapFig.layout));

        function renderLineChart(county) {{
            if (!countySeries[county]) return;

            Plotly.react(
                "line",
                countySeries[county],
                {{
                    title: "Load Profile and Prediction: " + county,
                    xaxis: {{
                        title: "Hour",
                        tickmode: "linear",
                        dtick: 4
                    }},
                    yaxis: {{
                        title: "Load (MW)"
                    }},
                    height: 700,
                    margin: {{l: 50, r: 20, t: 60, b: 50}},
                    legend: {{
                        orientation: "h",
                        y: -0.15
                    }}
                }},
                {{responsive: true}}
            );
        }}

        function renderMap(county=null) {{
            const layout = JSON.parse(JSON.stringify(baseMapLayout));
            let data = JSON.parse(JSON.stringify(baseMapData));

            if (county && countyLookup[county]) {{
                const info = countyLookup[county];

                layout.geo = layout.geo || {{}};
                layout.geo.center = {{
                    lat: info.lat,
                    lon: info.lon
                }};
                layout.geo.projection = layout.geo.projection || {{}};
                layout.geo.projection.scale = 18;

                const outlineTrace = {{
                    type: "choropleth",
                    geojson: data[0].geojson,
                    locations: [info.geoid],
                    z: [1],
                    featureidkey: "properties.GEOID",
                    colorscale: [[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                    showscale: false,
                    marker: {{
                        line: {{
                            color: "black",
                            width: 3
                        }}
                    }},
                    hoverinfo: "skip"
                }};

                data.push(outlineTrace);
            }}

            Plotly.react("map", data, layout, {{responsive: true}});
        }}

        function updateCountyView(county) {{
            if (!county || !countyLookup[county]) return;
            countyDropdown.value = county;
            renderLineChart(county);
            renderMap(county);
        }}

        mapDiv.on("plotly_click", function(eventData) {{
            const pt = eventData.points[0];
            const county = pt.customdata[0];
            updateCountyView(county);
        }});

        countyDropdown.addEventListener("change", function() {{
            const county = this.value;
            if (county) {{
                updateCountyView(county);
            }} else {{
                Plotly.react("line", lineFig.data, lineFig.layout, {{responsive: true}});
                renderMap();
            }}
        }});
    </script>
</body>
</html>
"""

output_path = f"graph/county_load_map/county_load_class_map_{today_str}.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_str)

print(f"Saved to: {output_path}")