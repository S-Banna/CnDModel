import os
import yaml
import json
import numpy as np
from PIL import Image
from collections import defaultdict
from datetime import datetime

def load_config():
    with open("../../data/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config["DATA_ROOT"]

# -------------------------
# DISASTER METADATA
# -------------------------
DISASTER_META = {
    "guatemala-volcano":    {"name": "Guatemala Fuego Volcano Eruption", "date": "Jun 3, 2018",              "tier": 1, "env": "Yes"},
    "hurricane-michael":    {"name": "Hurricane Michael",                 "date": "Oct 7-16, 2018",           "tier": 1, "env": "No"},
    "santa-rosa-wildfire":  {"name": "Santa Rosa Wildfires",              "date": "Oct 8-31, 2017",           "tier": 1, "env": "Yes"},
    "hurricane-florence":   {"name": "Hurricane Florence",                "date": "Sep 10-19, 2018",          "tier": 1, "env": "Yes"},
    "midwest-flooding":     {"name": "Midwest US Floods",                 "date": "Jan 3 - May 31, 2019",     "tier": 1, "env": "Yes"},
    "palu-tsunami":         {"name": "Indonesia Tsunami",                 "date": "Sep 18, 2018",             "tier": 1, "env": "Yes"},
    "socal-fire":           {"name": "Carr Wildfire",                     "date": "Jul 23 - Aug 30, 2018",    "tier": 1, "env": "No"},
    "hurricane-harvey":     {"name": "Hurricane Harvey",                  "date": "Aug 17 - Sep 2, 2017",     "tier": 1, "env": "No"},
    "mexico-earthquake":    {"name": "Mexico City Earthquake",            "date": "Sep 19, 2017",             "tier": 1, "env": "No"},
    "hurricane-matthew":    {"name": "Hurricane Matthew",                 "date": "Sep 28 - Oct 10, 2016",    "tier": 1, "env": "No"},
    "nepal-flooding":       {"name": "Monsoon in Nepal, India, Bangladesh","date": "Jul - Sep, 2017",         "tier": 1, "env": "Yes"},
    "moore-tornado":        {"name": "Moore, OK Tornado",                 "date": "May 20, 2013",             "tier": 3, "env": "No"},
    "tuscaloosa-tornado":   {"name": "Tuscaloosa, AL Tornado",            "date": "Apr 27, 2011",             "tier": 3, "env": "No"},
    "sunda-tsunami":        {"name": "Sunda Strait Tsunami",              "date": "Dec 22, 2018",             "tier": 3, "env": "No"},
    "lower-puna-volcano":   {"name": "Lower Puna Volcanic Eruption",      "date": "May 23 - Aug 14, 2018",    "tier": 3, "env": "Yes"},
    "joplin-tornado":       {"name": "Joplin, MO Tornado",                "date": "May 22, 2011",             "tier": 3, "env": "No"},
    "woolsey-fire":         {"name": "Woolsey Fire",                      "date": "Nov 9-28, 2018",           "tier": 3, "env": "No"},
    "pinery-bushfire":      {"name": "Pinery Fire",                       "date": "Nov 25 - Dec 2, 2018",     "tier": 3, "env": "No"},
    "portugal-wildfire":    {"name": "Portugal Wildfires",                "date": "Jun 17-24, 2017",          "tier": 3, "env": "No"},
}

def get_disaster_key(fname):
    # filenames like: guatemala-volcano_00000000_post_disaster.png
    # key is everything before the first underscore-number sequence
    parts = fname.split("_")
    # find where the numeric part starts
    for i, p in enumerate(parts):
        if p.isdigit():
            return "-".join(parts[:i])
    return fname

def main():
    DATA_ROOT = load_config()
    SUBSETS = ["tier1", "tier3", "hold"]

    # per disaster: total pairs, pairs with damage, subsets, capture dates
    stats = defaultdict(lambda: {
        "total": 0,
        "with_damage": 0,
        "subsets": set(),
        "capture_dates": [],
        "unmatched_prefix": False
    })

    unmatched_prefixes = set()

    for subset in SUBSETS:
        masks_dir  = os.path.join(DATA_ROOT, subset, "masks")
        images_dir = os.path.join(DATA_ROOT, subset, "images")

        if not os.path.exists(masks_dir):
            print(f"⚠️  Skipping {subset} — masks folder not found")
            continue

        post_masks = [
            f for f in os.listdir(masks_dir)
            if "_post" in f and "_rgb" not in f and f.endswith(".png")
        ]

        labels_dir = os.path.join(DATA_ROOT, subset, "labels")

        for fname in post_masks:
            key = get_disaster_key(fname)

            # check if we recognise this disaster
            if key not in DISASTER_META and not key.startswith("z-google-earth"):
                unmatched_prefixes.add(key)

            if key.startswith("z-google-earth"):
                continue  # skip our custom data

            mask = np.array(Image.open(os.path.join(masks_dir, fname)))
            has_damage = np.isin(mask, [3, 4]).any()

            stats[key]["total"] += 1
            stats[key]["subsets"].add(subset)
            if has_damage:
                stats[key]["with_damage"] += 1

            # pull capture date from json label if available
            if os.path.exists(labels_dir):
                json_fname = fname.replace(".png", ".json")
                json_path  = os.path.join(labels_dir, json_fname)
                if os.path.exists(json_path):
                    try:
                        with open(json_path) as f:
                            label = json.load(f)
                        date_str = label.get("metadata", {}).get("capture_date", "")
                        if date_str:
                            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                            stats[key]["capture_dates"].append(dt)
                    except Exception:
                        pass

    # -------------------------
    # PRINT TABLE
    # -------------------------
    col_w = [38, 22, 6, 5, 8, 14, 8, 26]
    header = (
        f"{'Disaster Event':<{col_w[0]}} "
        f"{'Event Dates':<{col_w[1]}} "
        f"{'Tier':<{col_w[2]}} "
        f"{'Env':<{col_w[3]}} "
        f"{'Subsets':<{col_w[4]}} "
        f"{'Total Pairs':>{col_w[5]}} "
        f"{'W/ Dmg':>{col_w[6]}} "
        f"{'Capture Date Range':<{col_w[7]}}"
    )
    divider = "-" * len(header)

    print("\n" + divider)
    print("xBD DATASET AUDIT — with vs without damage_only filter")
    print(divider)
    print(header)
    print(divider)

    total_pairs   = 0
    total_damaged = 0

    for key, meta in DISASTER_META.items():
        s = stats[key]
        subsets_str = "+".join(sorted(s["subsets"])) if s["subsets"] else "—"
        pct  = f"{100*s['with_damage']/s['total']:.0f}%" if s["total"] > 0 else "—"
        flag = " ⚠️" if s["with_damage"] == 0 else ""

        # capture date range
        if s["capture_dates"]:
            mn = min(s["capture_dates"]).strftime("%Y-%m-%d")
            mx = max(s["capture_dates"]).strftime("%Y-%m-%d")
            date_range = mn if mn == mx else f"{mn} → {mx}"
        else:
            date_range = "—"

        print(
            f"{meta['name']:<{col_w[0]}} "
            f"{meta['date']:<{col_w[1]}} "
            f"{meta['tier']:<{col_w[2]}} "
            f"{meta['env']:<{col_w[3]}} "
            f"{subsets_str:<{col_w[4]}} "
            f"{s['total']:>{col_w[5]}} "
            f"{s['with_damage']:>{col_w[6]}}  ({pct}){flag:4} "
            f"{date_range:<{col_w[7]}}"
        )
        total_pairs   += s["total"]
        total_damaged += s["with_damage"]

    print(divider)
    print(
        f"{'TOTAL':<{col_w[0]}} "
        f"{'':<{col_w[1]}} "
        f"{'':<{col_w[2]}} "
        f"{'':<{col_w[3]}} "
        f"{'':<{col_w[4]}} "
        f"{total_pairs:>{col_w[5]}} "
        f"{total_damaged:>{col_w[6]}}"
    )
    print(divider)

    print(f"\ndamage_only filter retains {total_damaged}/{total_pairs} pairs "
          f"({100*total_damaged/total_pairs:.1f}%) across all subsets.\n")

    if unmatched_prefixes:
        print(f"⚠️  Unrecognised disaster prefixes (not in metadata table):")
        for p in sorted(unmatched_prefixes):
            print(f"   {p}")

if __name__ == "__main__":
    main()