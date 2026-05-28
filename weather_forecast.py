import time
import requests
import pandas as pd
from datetime import datetime

zone_weather = [
    ["AE", "ACY", 100.0],
    ["AEP", "CAK", 15.1],
    ["AEP", "CMH", 23.4],
    ["AEP", "CRW", 22.6],
    ["AEP", "FWA", 22.7],
    ["AEP", "ROA", 16.2],
    ["APS", "IAD", 30.0],
    ["APS", "PIT", 70.0],
    ["ATSI", "CAK", 46.5],
    ["ATSI", "CLE", 30.0],
    ["ATSI", "PIT", 8.5],
    ["ATSI", "TOL", 15.0],
    ["BGE", "BWI", 100.0],
    ["COMED", "ORD", 100.0],
    ["DAYTON", "DAY", 100.0],
    ["DPL", "ILG", 70.0],
    ["DPL", "WAL", 30.0],
    ["DQE", "PIT", 100.0],
    ["JCPL", "ACY", 25.0],
    ["JCPL", "EWR", 75.0],
    ["METED", "ABE", 50.0],
    ["METED", "PHL", 50.0],
    ["PECO", "PHL", 100.0],
    ["PENLC", "ERI", 50.0],
    ["PENLC", "IPT", 50.0],
    ["PEPCO", "DCA", 100.0],
    ["PL", "ABE", 25.0],
    ["PL", "AVP", 25.0],
    ["PL", "IPT", 25.0],
    ["PL", "MDT", 25.0],
    ["PS", "EWR", 100.0],
    ["RECO", "EWR", 100.0],
    ["UGI", "AVP", 100.0],
    ["VEPCO", "IAD", 33.3],
    ["VEPCO", "ORF", 33.3],
    ["VEPCO", "RIC", 33.3],
    ["DUKE", "CVG", 100.0],
    ["EKPC", "CVG", 25.0],
    ["EKPC", "LEX", 49.0],
    ["EKPC", "SDF", 26.0],
]

zone_weather = pd.DataFrame(zone_weather, columns=["zone", "weather_station", "weather_weight"])
zone_weather["weather_weight"] = zone_weather["weather_weight"] / 100

# your airport list
airports = [
'ACY','CAK','CMH','CRW','FWA','ROA','IAD','PIT','CAK','CLE','PIT','TOL',
'BWI','ORD','DAY','ILG','WAL','PIT','ACY','EWR','ABE','PHL','PHL','ERI',
'IPT','DCA','ABE','AVP','IPT','MDT','EWR','EWR','AVP','IAD','ORF','RIC',
'CVG','CVG','LEX','SDF'
]

# download airport database
url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

cols = [
"airport_id","name","city","country","iata","icao",
"lat","lon","altitude","timezone","dst","tz","type","source"
]

df_airports = pd.read_csv(url, header=None, names=cols)

airport_locations = df_airports[df_airports["iata"].isin(set(airports))][["iata","lat","lon"]]

zone_weather = zone_weather.merge(
    airport_locations,
    left_on="weather_station",
    right_on="iata",
    how="left"
)

zone_weather = zone_weather.drop(columns="iata")

zone_weather["lat"] = zone_weather["lat"].round(2)
zone_weather["lon"] = zone_weather["lon"].round(2)


start_time = pd.Timestamp.now().normalize() # "2026-01-01 00:00:00"
end_time = pd.Timestamp.now().normalize() + pd.Timedelta(days=7) # "2026-03-31 23:00:00"

# define time range
time_index = pd.date_range(
    start= start_time,
    end= end_time,
    freq="h"
)

# cross join
zone_weather["key"] = 1
time_df = pd.DataFrame({"time": time_index, "key": 1})

zone_weather = zone_weather.merge(time_df, on="key").drop("key", axis=1)


df = zone_weather.copy()

HOURLY_VARS = [
    # Temperature
    "temperature_2m",  # vital
    # "temperature_80m",
    "apparent_temperature",
    "dew_point_2m",
    # "soil_temperature_0cm",
    # "soil_temperature_6cm",

    # Humidity
    "relative_humidity_2m",  # vital

    # Precipitation
    # "precipitation_probability",
    "precipitation",  # vital
    "rain",
    # "showers",
    "snowfall",
    "snow_depth",

    # Cloud
    "cloud_cover",  # vital
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",

    # Wind
    "wind_speed_10m",  # vital
    # "wind_speed_80m",
    "wind_direction_10m",
    # "wind_direction_80m",
    "wind_gusts_10m",

    # Pressure
    "surface_pressure",
    "pressure_msl",

    # # Soil moisture
    # "soil_moisture_0_to_1cm",
    # "soil_moisture_1_to_3cm",

    # Weather codes
    "weather_code",

    # # Evapotranspiration
    # "visibility",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",
    # "evapotranspiration",

    # Radiation
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "global_tilted_irradiance",
    "direct_normal_irradiance",
    "terrestrial_radiation",

    # # Other
    # "uv_index",
    # "is_day",
    # "uv_index_clear_sky",
    # "boundary_layer_height",
    # "sunshine_duration",
    # "wet_bulb_temperature_2m",
    # "total_column_integrated_water_vapour",
    # "cape", 
    # "lifted_index", 
    # "convective_inhibition", 
    # "freezing_level_height", 
    # "boundary_layer_height"
]

ARCHIVE_URL = "https://customer-api.open-meteo.com/v1/forecast"

def month_ranges(start_ts, end_ts):
    """
    Split a datetime range into monthly [start_date, end_date] tuples as strings.
    """
    start_ts = pd.Timestamp(start_ts).normalize()
    end_ts = pd.Timestamp(end_ts).normalize()

    month_starts = pd.date_range(
        start=start_ts.replace(day=1),
        end=end_ts.replace(day=1),
        freq="MS"
    )

    ranges = []
    for ms in month_starts:
        me = ms + pd.offsets.MonthEnd(0)
        chunk_start = max(start_ts, ms)
        chunk_end = min(end_ts, me)
        ranges.append((chunk_start.date().isoformat(), chunk_end.date().isoformat()))
    return ranges


def fetch_hourly_weather_chunk(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_retries: int = 2,
    base_sleep: float = 2.0,
):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,   # YYYY-MM-DD
        "end_date": end_date,       # YYYY-MM-DD
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/New_York",
        "timeformat": "iso8601",
        "apikey": "2a6seG5JAxNTJQQi",
    }

    for attempt in range(max_retries):
        r = requests.get(ARCHIVE_URL, params=params, timeout=60)

        if r.status_code == 429:
            wait = base_sleep * (2 ** attempt)
            print(
                f"429 for ({lat}, {lon}) [{start_date} to {end_date}] "
                f"- retry {attempt + 1}/{max_retries}, sleep {wait:.1f}s"
            )
            time.sleep(wait)
            continue

        r.raise_for_status()
        data = r.json()

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])

        if not times:
            return pd.DataFrame(columns=["time"] + HOURLY_VARS)

        out = pd.DataFrame({"time": pd.to_datetime(times)})
        for v in HOURLY_VARS:
            out[v] = hourly.get(v)

        return out

    raise RuntimeError(
        f"Exceeded retries for ({lat}, {lon}) [{start_date} to {end_date}]"
    )


weather_frames = []

for (lat, lon), g in df.groupby(["lat", "lon"], sort=False):
    loc_start = g["time"].min()
    loc_end = g["time"].max()

    monthly_windows = month_ranges(loc_start, loc_end)
    location_chunks = []

    print(f"Fetching ({lat}, {lon}) in {len(monthly_windows)} monthly chunks...")

    for start_date, end_date in monthly_windows:
        try:
            w = fetch_hourly_weather_chunk(lat, lon, start_date, end_date)
            if not w.empty:
                w["lat"] = lat
                w["lon"] = lon
                location_chunks.append(w)

            # small throttle between requests
            time.sleep(0.01)

        except Exception as e:
            print(f"Failed for ({lat}, {lon}) [{start_date} to {end_date}]: {e}")

    if location_chunks:
        weather_frames.append(pd.concat(location_chunks, ignore_index=True))

if weather_frames:
    weather_df = pd.concat(weather_frames, ignore_index=True)
    weather_df = weather_df.drop_duplicates(subset=["lat", "lon", "time"])
else:
    weather_df = pd.DataFrame(columns=["time", "lat", "lon"] + HOURLY_VARS)

# Make sure merge keys have matching datetime dtype
df["time"] = pd.to_datetime(df["time"])
weather_df["time"] = pd.to_datetime(weather_df["time"])

# Merge back on time + lat + lon
enriched_df = df.merge(
    weather_df,
    on=["lat", "lon", "time"],
    how="left",
    validate="many_to_one",)

today_str = datetime.today().strftime('%y%m%d')

enriched_df.to_csv(f'data/weather/weather_forecast_{today_str}.csv', index=False)












































