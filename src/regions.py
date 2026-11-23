"""Group sightings into Kenyan habitats/regions from their GPS coordinates (numpy only).

Each sighting is assigned to the nearest hub within MAX_KM, otherwise "Elsewhere".
Hub coordinates are approximate park centres, enough to separate broad habitats:
open grassland (Mara), semi-arid savanna (Amboseli, Tsavo, Samburu), lake/woodland (Nakuru-Naivasha)
and the urban-edge Nairobi National Park.
"""
import numpy as np

HUBS = {
    "Maasai Mara": (-1.50, 35.14),
    "Nakuru-Naivasha": (-0.55, 36.25),
    "Nairobi NP": (-1.37, 36.80),
    "Amboseli": (-2.65, 37.26),
    "Tsavo": (-3.00, 38.45),
    "Laikipia-Samburu": (0.45, 37.20),
}
MAX_KM = 120


def _haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def assign_regions(lat, lon, max_km=MAX_KM):
    lat, lon = np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)
    names = list(HUBS)
    dist = np.stack([_haversine(lat, lon, la, lo) for la, lo in HUBS.values()], axis=1)
    nearest = dist.argmin(axis=1)
    out = np.array([names[i] if dist[k, i] <= max_km else "Elsewhere" for k, i in enumerate(nearest)], dtype=object)
    out[np.isnan(lat) | np.isnan(lon)] = "Unknown"
    return out
