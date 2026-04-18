"""Weather lookup tool.

Fetches current weather for a city using the open-meteo.com free API.
No API key, no signup, no rate limit for hobby usage.

Implementation is a two-step call:
  1. Geocoding endpoint — city name -> latitude/longitude
  2. Forecast endpoint — current weather at those coordinates
"""
import json
import logging
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes — https://open-meteo.com/en/docs (§ Weather Variable Documentation)
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def _http_get_json(url: str) -> dict:
    """GET a URL and return the parsed JSON response."""
    req = urllib.request.Request(url, headers={"User-Agent": "AgentForge/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _geocode(city: str):
    """Resolve a city name to (latitude, longitude, display_name). Returns None on miss."""
    params = urllib.parse.urlencode({
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
    })
    data = _http_get_json(f"{GEOCODE_URL}?{params}")
    results = data.get("results") or []
    if not results:
        return None
    r = results[0]
    country = r.get("country", "")
    display = f"{r['name']}, {country}" if country else r["name"]
    return r["latitude"], r["longitude"], display


def _c_to_f(celsius: float) -> int:
    return round(celsius * 9 / 5 + 32)


def get_weather(city: str) -> str:
    """Return current weather for a city as a compact one-line string."""
    logger.info("Weather lookup invoked for city: %s", city)
    try:
        if not city or not isinstance(city, str):
            return "Error: city must be a non-empty string"

        geo = _geocode(city.strip())
        if geo is None:
            return f"No weather data found for '{city}' (city not recognized)"
        lat, lon, display = geo

        params = urllib.parse.urlencode({
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,weather_code,wind_speed_10m",
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
        })
        data = _http_get_json(f"{FORECAST_URL}?{params}")
        current = data.get("current", {})
        temp_c = current.get("temperature_2m")
        code = current.get("weather_code")
        wind = current.get("wind_speed_10m")

        if temp_c is None or code is None:
            return f"Weather data incomplete for '{display}'"

        description = WEATHER_CODES.get(int(code), f"weather code {code}")
        temp_f = _c_to_f(temp_c)
        wind_str = f", wind {round(wind)} km/h" if wind is not None else ""

        return f"{display}: {round(temp_c)}°C ({temp_f}°F), {description}{wind_str}"

    except urllib.error.HTTPError as e:
        return f"Weather API error (HTTP {e.code}) for '{city}'"
    except urllib.error.URLError as e:
        return f"Could not reach weather service: {e.reason}"
    except Exception as e:
        return f"Error looking up weather for '{city}': {e}"


TOOL_FUNCTION = get_weather

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather (temperature, conditions, wind) for a city by name",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Tokyo', 'New York', 'São Paulo'",
                }
            },
            "required": ["city"],
        },
    },
}
