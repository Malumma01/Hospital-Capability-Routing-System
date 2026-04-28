# utils.py

import random
import time
from math import radians, sin, cos, atan2, sqrt

# ---------------------------------------------------------
# Distance calculation (Haversine formula)
# ---------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates in kilometers.
    """
    R = 6371  # Earth radius in km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)

    a = sin(dphi/2)**2 + cos(phi1) * cos(phi2) * sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


# ---------------------------------------------------------
# Real-time hospital status simulation
# ---------------------------------------------------------
def simulate_status(row):
    """
    Simulate hospital availability status.
    Status changes every minute.
    """
    seed_value = int(time.time() // 60) + int(row["lat"] * 1000)
    random.seed(seed_value)

    return random.choices(
        ["Available", "Busy", "Offline"],
        weights=[0.6, 0.3, 0.1],
        k=1
    )[0]


# ---------------------------------------------------------
# Ambulance ETA estimation
# ---------------------------------------------------------
def estimate_ambulance_eta(distance_km):
    """
    Estimate ambulance arrival time based on distance.
    Assumes average speed of 30 km/h in city traffic.
    """
    if distance_km <= 0.5:
        return 5  # minutes

    return int((distance_km / 30) * 60) + 5


# ---------------------------------------------------------
# Hospital ranking logic
# ---------------------------------------------------------
def rank_hospitals(df, user_lat, user_lon):
    """
    Add distance and status columns, then rank hospitals.
    """
    df = df.copy()

    # Compute distance
    df["distance_km"] = df.apply(
        lambda row: haversine(user_lat, user_lon, row["lat"], row["lon"]),
        axis=1
    )

    # Simulate real-time status
    df["status"] = df.apply(simulate_status, axis=1)

    # Rank by status then distance
    status_order = {"Available": 0, "Busy": 1, "Offline": 2}
    df["status_rank"] = df["status"].map(status_order)

    df = df.sort_values(["status_rank", "distance_km"])

    return df