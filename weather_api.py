# pip install openmeteo-requests
# pip install requests-cache retry-requests numpy pandas

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# CONFIG
API_KEY = "2a6seG5JAxNTJQQi"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://customer-api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    # Temperature
    "temperature_2m",
    "apparent_temperature",
    "dew_point_2m",

    # Humidity
    "relative_humidity_2m",

    # Precipitation
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",

    # Cloud
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",

    # Wind
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",

    # Pressure
    "surface_pressure",
    "pressure_msl",

    # Weather codes
    "weather_code",

    # Evapotranspiration
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit",

    # Radiation
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "global_tilted_irradiance",
    "direct_normal_irradiance",
    "terrestrial_radiation",
]


# CLIENT
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def build_hourly_dataframe(response, requested_vars):
    """
    Convert one Open-Meteo response into a pandas DataFrame.
    Assumes variable order matches requested_vars order.
    """
    hourly = response.Hourly()

    df = pd.DataFrame({
        "datetime_utc": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
    })

    for i, var_name in enumerate(requested_vars):
        df[var_name] = hourly.Variables(i).ValuesAsNumpy()

    return df


def fetch_archive(lat, lon, start_date, end_date, hourly_vars):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
    }

    responses = openmeteo.weather_api(ARCHIVE_URL, params=params)
    response = responses[0]
    df = build_hourly_dataframe(response, hourly_vars)
    df["source"] = "archive"
    return df


def fetch_forecast(lat, lon, hourly_vars):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": hourly_vars,
        "apikey": API_KEY,
    }

    responses = openmeteo.weather_api(FORECAST_URL, params=params)
    response = responses[0]
    df = build_hourly_dataframe(response, hourly_vars)
    df["source"] = "forecast"
    return df


def combine_archive_forecast(df_archive, df_forecast):
    """
    Keep archive for historical timestamps.
    Use forecast for future timestamps.
    If there is overlap, keep archive first, then forecast for remaining gaps.
    """
    combined = pd.concat([df_archive, df_forecast], ignore_index=True)

    # Prefer archive over forecast on overlapping timestamps
    source_priority = {"archive": 0, "forecast": 1}
    combined["source_priority"] = combined["source"].map(source_priority)

    combined = (
        combined
        .sort_values(["datetime_utc", "source_priority"])
        .drop_duplicates(subset=["datetime_utc"], keep="first")
        .drop(columns=["source_priority"])
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )

    return combined


# EXAMPLE

# LATITUDE = 52.52
# LONGITUDE = 13.41

# ARCHIVE_START = "2026-03-18"
# ARCHIVE_END   = "2026-04-01"


# archive_df = fetch_archive(
#     lat=LATITUDE,
#     lon=LONGITUDE,
#     start_date=ARCHIVE_START,
#     end_date=ARCHIVE_END,
#     hourly_vars=HOURLY_VARS,
# )

# forecast_df = fetch_forecast(
#     lat=LATITUDE,
#     lon=LONGITUDE,
#     hourly_vars=HOURLY_VARS,
# )

# weather_df = combine_archive_forecast(archive_df, forecast_df)

