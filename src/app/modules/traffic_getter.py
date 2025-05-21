import requests

def get_road_from_gps(latitude, longitude):
    """
    Uses OpenStreetMap's Nominatim API to get the road name for given GPS coordinates.
    Returns the road name or None if not found.
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "json",
        "addressdetails": 1
    }
    headers = {
        "User-Agent": "RoadDefectIndexingSystem/1.0"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        address = data.get("address", {})
        # Try to get the most specific road-related field
        for key in ["road", "pedestrian", "footway", "cycleway", "path"]:
            if key in address:
                return address[key]
    return None

def get_traffic_volume_from_road(road_name: str) -> int:
    """
    Returns the traffic volume for a given road name.
    If the road is not recognized, returns a default value.
    """
    road_traffic = {
        "C-4 (EDSA)": 419_952,
        "C-5 (E. Rodriguez Jr. Ave/C.P. Garcia)": 250_782,
        "Ortigas Avenue (R-5)": 188_765,
        "Shaw Boulevard (R-4)": 144_936,
        # Add more mappings if needed
    }
    # Try exact match first
    if road_name in road_traffic:
        return road_traffic[road_name]
    # Try partial match (case-insensitive)
    for key in road_traffic:
        if key.lower() in road_name.lower():
            return road_traffic[key]
    return 50000  # Default value

def get_road_and_traffic_from_gps(latitude, longitude):
    """
    Returns (road_name, traffic_volume) for given GPS coordinates.
    """
    road_name = get_road_from_gps(latitude, longitude)
    if road_name:
        traffic = get_traffic_volume_from_road(road_name)
        return road_name, traffic
    return None, 50000