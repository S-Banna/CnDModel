import os
import json
import yaml

# -----------------------------
# LOAD CONFIG
# -----------------------------
def load_config():
    config_path = "../../data/config.yaml" 
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["DATA_ROOT"]


# -----------------------------
# CLASSIFY BUILDINGS
# -----------------------------
def classify_buildings(data_root): # Scans all JSONs and counts buildings per subtype
    subtype_counts = {}
    binary_counts = {"damaged": 0, "undamaged": 0}

    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith("_post_disaster.json"):
                json_path = os.path.join(root, file)
                with open(json_path) as f:
                    data = json.load(f)

                for feat in data["features"]["lng_lat"]:
                    subtype = feat["properties"]["subtype"]
                    subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1

                    # binary collapse metrics
                    if subtype in ["major-damage", "destroyed"]:
                        binary_counts["damaged"] += 1
                    elif subtype in ["no-damage", "minor-damage"]:
                        binary_counts["undamaged"] += 1
                    else:
                        # unclassified or unknown
                        pass

    return subtype_counts, binary_counts


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    DATA_ROOT = load_config()
    print(f"Scanning dataset at: {DATA_ROOT}\n")

    subtype_counts, binary_counts = classify_buildings(DATA_ROOT + "/labels")

    print("Subtype counts:")
    for k, v in subtype_counts.items():
        print(f"{k}: {v}")

    print("\nBinary damage counts:")
    for k, v in binary_counts.items():
        print(f"{k}: {v}")
