"""
location_enricher.py
====================
Given a place name (or lat/lon), automatically fetches:
  - Coordinates & elevation  → Open-Meteo Elevation API  (free, no key)
  - Current + 7-day rainfall → Open-Meteo Weather API    (free, no key)
  - Reverse geocoding        → Nominatim / OpenStreetMap  (free, no key)
  - River proximity estimate → OpenTopoData / heuristic   (free, no key)

All APIs used are completely free and require no API key.
"""

import math
import time
import requests

# ── constants ────────────────────────────────────────────────────
NOMINATIM_URL   = "https://nominatim.openstreetmap.org/search"
GEOCODE_REVERSE = "https://nominatim.openstreetmap.org/reverse"
ELEVATION_URL   = "https://api.open-meteo.com/v1/elevation"
WEATHER_URL     = "https://api.open-meteo.com/v1/forecast"
HEADERS         = {"User-Agent": "FloodRiskMLOpsidian/1.0 (educational project)"}

# Sri Lanka district boundaries (fallback mapping for common places)
SRI_LANKA_DISTRICTS = {
    "colombo": "Colombo", "gampaha": "Gampaha", "kalutara": "Kalutara",
    "kandy": "Kandy", "matale": "Matale", "nuwara eliya": "Nuwara Eliya",
    "galle": "Galle", "matara": "Matara", "hambantota": "Hambantota",
    "jaffna": "Jaffna", "kilinochchi": "Kilinochchi", "mannar": "Mannar",
    "vavuniya": "Vavuniya", "mullaitivu": "Mullaitivu", "batticaloa": "Batticaloa",
    "ampara": "Ampara", "trincomalee": "Trincomalee", "kurunegala": "Kurunegala",
    "puttalam": "Puttalam", "anuradhapura": "Anuradhapura", "polonnaruwa": "Polonnaruwa",
    "badulla": "Badulla", "moneragala": "Moneragala", "ratnapura": "Ratnapura",
    "kegalle": "Kegalle",
}

def _safe_get(url, params, timeout=10):
    """HTTP GET with error handling."""
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None


def geocode_place(place_name: str) -> dict:
    """
    Convert a place name → {lat, lon, display_name, district, exact_place, country}.
    Returns None on failure.

    Uses Nominatim's structured address to extract both the exact place
    (neighbourhood / suburb / quarter / amenity / road) AND the district,
    so entering 'Colombo' returns 'Colombo' as exact_place, not just the
    district boundary centroid.
    """
    data = _safe_get(NOMINATIM_URL, {
        "q": place_name,
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    })
    if not data:
        return None
    hit = data[0]
    addr = hit.get("address", {})

    # ── Exact place: most specific named feature available ────────
    exact_place = (
        addr.get("amenity") or
        addr.get("tourism") or
        addr.get("leisure") or
        addr.get("neighbourhood") or
        addr.get("suburb") or
        addr.get("quarter") or
        addr.get("city_district") or
        addr.get("hamlet") or
        addr.get("road") or
        addr.get("village") or
        addr.get("town") or
        addr.get("city") or
        place_name
    )

    # ── District: administrative area ────────────────────────────
    district = (
        addr.get("district") or
        addr.get("county") or
        addr.get("state_district") or
        addr.get("city") or
        addr.get("town") or
        addr.get("village") or
        ""
    )

    # Try to match against known Sri Lanka districts
    district_clean = district.strip().lower()
    for key, val in SRI_LANKA_DISTRICTS.items():
        if key in district_clean or district_clean in key:
            district = val
            break

    return {
        "lat": float(hit["lat"]),
        "lon": float(hit["lon"]),
        "display_name": hit.get("display_name", place_name),
        "district": district,
        "exact_place": exact_place,
        "country": addr.get("country", ""),
        # city is used to pre-fill the Place Name field
        "city": exact_place,
    }


def get_elevation(lat: float, lon: float) -> float | None:
    """Fetch elevation in metres from Open-Meteo elevation API."""
    data = _safe_get(ELEVATION_URL, {"latitude": lat, "longitude": lon})
    if data and "elevation" in data:
        elev = data["elevation"]
        if isinstance(elev, list):
            return float(elev[0])
        return float(elev)
    return None


def get_weather_data(lat: float, lon: float) -> dict:
    """
    Fetch weather from Open-Meteo:
      - precipitation_sum for past 7 days  → rainfall_7d_mm
      - precipitation_sum for past 30 days → monthly_rainfall_mm
      - extreme_weather_index (derived)
      - seasonal_index (derived from month)
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "precipitation_sum",
            "rain_sum",
            "precipitation_probability_max",
            "windspeed_10m_max",
        ],
        "past_days": 30,
        "forecast_days": 1,
        "timezone": "auto",
    }
    data = _safe_get(WEATHER_URL, params)
    result = {}

    if data and "daily" in data:
        daily = data["daily"]
        precip = daily.get("precipitation_sum", [])
        precip = [p if p is not None else 0.0 for p in precip]

        # 7-day and 30-day totals (last N days from list)
        result["rainfall_7d_mm"]      = round(sum(precip[-7:]),  2) if len(precip) >= 7  else round(sum(precip), 2)
        result["monthly_rainfall_mm"] = round(sum(precip[-30:]), 2) if len(precip) >= 30 else round(sum(precip), 2)

        # Extreme weather index: normalise by heavy-rain threshold (~100mm/week)
        ew = min(result["rainfall_7d_mm"] / 100.0, 1.0)
        wind_max = daily.get("windspeed_10m_max", [])
        wind_max = [w for w in wind_max if w is not None]
        if wind_max:
            wind_factor = min(max(wind_max) / 80.0, 1.0)   # 80 km/h → extreme
            ew = round((ew * 0.7 + wind_factor * 0.3), 4)
        result["extreme_weather_index"] = round(ew, 4)

        # Seasonal index: monsoon months for South/SE Asia get higher index
        import datetime
        month = datetime.datetime.now().month
        # SW monsoon: May–Sep → high; NE monsoon: Oct–Jan → moderate
        seasonal_map = {1: 0.7, 2: 0.3, 3: 0.3, 4: 0.4, 5: 0.7,
                        6: 0.9, 7: 0.9, 8: 0.8, 9: 0.7, 10: 0.6,
                        11: 0.7, 12: 0.6}
        result["seasonal_index"] = seasonal_map.get(month, 0.5)

        # Water presence flag (heuristic)
        result["water_presence_flag"] = "Likely" if result["rainfall_7d_mm"] > 30 else "Unlikely"

    return result


def estimate_distance_to_river(lat: float, lon: float, elevation: float | None) -> dict:
    """
    Estimate distance to nearest river using elevation as proxy:
    - Very low elevation (< 5m)  → likely near coast/river → ~100–300m
    - Low elevation (5–30m)      → moderate proximity     → ~500–1500m
    - Medium (30–100m)           → further                → ~1500–5000m
    - High (>100m)               → typically far          → 5000–20000m

    Also derives terrain_roughness_index and drainage_index from elevation.
    """
    result = {}

    if elevation is not None:
        if elevation < 5:
            dist = 150.0
            drainage = 0.3   # poor drainage at very low elevation
            terrain  = 0.2
        elif elevation < 30:
            dist = 800.0
            drainage = 0.5
            terrain  = 0.35
        elif elevation < 100:
            dist = 2500.0
            drainage = 0.65
            terrain  = 0.55
        else:
            dist = 8000.0
            drainage = 0.8
            terrain  = 0.7

        result["distance_to_river_m"]   = dist
        result["drainage_index"]         = drainage
        result["terrain_roughness_index"] = terrain
    else:
        result["distance_to_river_m"]   = 1000.0
        result["drainage_index"]         = 0.5
        result["terrain_roughness_index"] = 0.4

    return result


def enrich_location(place_name: str = None, lat: float = None, lon: float = None) -> dict:
    """
    Master function: given a place name and/or coordinates, return a dict
    with all auto-fetched feature values ready to pre-fill the prediction form.

    Returns a dict with keys matching PredictRequest fields, plus
    a 'meta' key with status/messages for display.
    """
    result = {"meta": {"sources": [], "warnings": []}}

    # ── Step 1: Geocoding ────────────────────────────────────────
    if place_name and (lat is None or lon is None):
        geo = geocode_place(place_name)
        if geo:
            lat = geo["lat"]
            lon = geo["lon"]
            result["latitude"]  = lat
            result["longitude"] = lon
            result["place_name"] = geo["city"]
            if geo["district"]:
                result["district"] = geo["district"]
            result["meta"]["display_name"] = geo["display_name"]
            result["meta"]["sources"].append("📍 Coordinates from OpenStreetMap/Nominatim")
        else:
            result["meta"]["warnings"].append(f"⚠️ Could not geocode '{place_name}'. Try a more specific name.")
            return result
    elif lat is not None and lon is not None:
        result["latitude"]  = lat
        result["longitude"] = lon
        # Reverse geocode to get place details
        rev = _safe_get(GEOCODE_REVERSE, {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1})
        if rev:
            addr = rev.get("address", {})
            district = (addr.get("district") or addr.get("county") or
                        addr.get("state_district") or addr.get("city") or "")
            for key, val in SRI_LANKA_DISTRICTS.items():
                if key in district.lower() or district.lower() in key:
                    district = val
                    break
            if district:
                result["district"] = district
            # Use most specific place name available
            exact_place = (
                addr.get("amenity") or addr.get("tourism") or
                addr.get("neighbourhood") or addr.get("suburb") or
                addr.get("quarter") or addr.get("city_district") or
                addr.get("hamlet") or addr.get("road") or
                addr.get("village") or addr.get("town") or
                addr.get("city") or place_name or ""
            )
            if exact_place:
                result["place_name"] = exact_place
            result["meta"]["display_name"] = rev.get("display_name", f"{lat}, {lon}")
            result["meta"]["sources"].append("📍 Location details from OpenStreetMap/Nominatim")

    if lat is None or lon is None:
        result["meta"]["warnings"].append("⚠️ No valid coordinates — cannot fetch environmental data.")
        return result

    # ── Step 2: Elevation ────────────────────────────────────────
    time.sleep(0.2)   # be polite to free APIs
    elevation = get_elevation(lat, lon)
    if elevation is not None:
        result["elevation_m"] = round(elevation, 1)
        result["meta"]["sources"].append("🏔️ Elevation from Open-Meteo Elevation API")
    else:
        result["meta"]["warnings"].append("⚠️ Could not fetch elevation data.")

    # ── Step 3: Weather / Rainfall ───────────────────────────────
    time.sleep(0.2)
    weather = get_weather_data(lat, lon)
    if weather:
        result.update(weather)
        result["meta"]["sources"].append("🌧️ Rainfall & weather from Open-Meteo Weather API (30-day history)")
    else:
        result["meta"]["warnings"].append("⚠️ Could not fetch weather data.")

    # ── Step 4: River distance estimate ──────────────────────────
    river_data = estimate_distance_to_river(lat, lon, elevation)
    result.update(river_data)
    result["meta"]["sources"].append("🏞️ River distance & terrain estimated from elevation profile")

    # ── Step 5: Land-use heuristics from coordinates ─────────────
    # For Sri Lanka: coastal (very low elev) → high built-up, surface water risk
    elev = elevation or 50.0
    if elev < 10:
        result.setdefault("built_up_percent",  55.0)
        result.setdefault("ndwi",              0.25)
        result.setdefault("ndvi",              0.15)
        result.setdefault("water_supply",      "Surface water")
        result.setdefault("flood_occurrence_current_event", "Yes" if weather.get("rainfall_7d_mm", 0) > 50 else "No")
    elif elev < 50:
        result.setdefault("built_up_percent",  35.0)
        result.setdefault("ndwi",              0.10)
        result.setdefault("ndvi",              0.30)
        result.setdefault("water_supply",      "Municipal")
        result.setdefault("flood_occurrence_current_event", "Yes" if weather.get("rainfall_7d_mm", 0) > 80 else "No")
    else:
        result.setdefault("built_up_percent",  20.0)
        result.setdefault("ndwi",             -0.05)
        result.setdefault("ndvi",              0.50)
        result.setdefault("water_supply",      "Well")
        result.setdefault("flood_occurrence_current_event", "No")

    result["meta"]["sources"].append("🌿 Vegetation & land-use estimated from elevation profile")

    return result
