STRUCTURES = {
    "Residential Low Rise": {
        "floors": 2,
        "splits": {"concrete": 0.60, "steel": 0.10, "masonry": 0.30},
    },
    "Residential High Rise": {
        "floors": 5,
        "splits": {"concrete": 0.60, "steel": 0.10, "masonry": 0.30},
    },
    "Industrial": {
        "floors": 1,
        "splits": {"concrete": 0.40, "steel": 0.40, "masonry": 0.20},
    },
}

DENSITIES = {
    "concrete": 2400,
    "masonry":  1800,
    "steel":    7850,
}

FLOOR_HEIGHT    = 3.0   # metres per floor
COLLAPSE_FACTOR = 0.6   # fraction of total volume that becomes rubble


def quantify_building(building, gsd, structure_type):
    config  = STRUCTURES[structure_type]
    floors  = config["floors"]

    area_m2   = building["pixel_area"] * (gsd ** 2)
    height_m  = floors * FLOOR_HEIGHT
    volume_m3 = area_m2 * height_m

    # Only the collapsed fraction contributes to rubble mass
    rubble_volume_m3 = volume_m3 * COLLAPSE_FACTOR

    concrete_kg = masonry_kg = steel_kg = 0.0

    for material, percent in config["splits"].items():
        mass = rubble_volume_m3 * percent * DENSITIES[material]
        if material == "concrete":
            concrete_kg = mass
        elif material == "masonry":
            masonry_kg  = mass
        elif material == "steel":
            steel_kg    = mass

    return {
        "building_id":    building["id"],
        "pixel_area":     building["pixel_area"],
        "area_m2":        round(area_m2, 2),
        "height_m":       round(height_m, 2),
        "volume_m3":      round(volume_m3, 2),
        "rubble_volume_m3": round(rubble_volume_m3, 2),
        "collapse_factor":  COLLAPSE_FACTOR,
        "concrete_kg":    round(concrete_kg, 2),
        "masonry_kg":     round(masonry_kg, 2),
        "steel_kg":       round(steel_kg, 2),
        "mass_kg":        round(concrete_kg + masonry_kg + steel_kg, 2),
    }