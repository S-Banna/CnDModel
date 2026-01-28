import json
import cv2
import numpy as np
from shapely import wkt

# ---- paths ----
IMAGE_PATH = "guatemala-volcano_00000023_pre_disaster.tif"      # or .png / .jpg
JSON_PATH = "guatemala-volcano_00000023_pre_disaster.json"

# ---- load image ----
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise RuntimeError("Could not load image")

h, w, _ = img.shape

# ---- load json ----
with open(JSON_PATH, "r") as f:
    data = json.load(f)

features = data["features"]["lng_lat"]

# ---- collect all lon/lat points to normalize ----
all_coords = []

polygons = []
for feat in features:
    poly = wkt.loads(feat["wkt"])
    coords = np.array(poly.exterior.coords)
    polygons.append(coords)
    all_coords.append(coords)

all_coords = np.vstack(all_coords)

lon_min, lat_min = all_coords.min(axis=0)
lon_max, lat_max = all_coords.max(axis=0)

# ---- helper: lon/lat → pixel ----
def geo_to_pixel(coords):
    lon = coords[:, 0]
    lat = coords[:, 1]

    x = (lon - lon_min) / (lon_max - lon_min) * w
    y = (lat_max - lat) / (lat_max - lat_min) * h  # flip y

    return np.stack([x, y], axis=1).astype(np.int32)

# ---- draw polygons ----
overlay = img.copy()

for coords in polygons:
    pts = geo_to_pixel(coords).reshape((-1, 1, 2))
    cv2.polylines(overlay, [pts], True, (0, 0, 255), 1)
    cv2.fillPoly(overlay, [pts], (0, 0, 255))

vis = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

# ---- show ----
cv2.imshow("Damage Polygons (Demo Overlay)", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()