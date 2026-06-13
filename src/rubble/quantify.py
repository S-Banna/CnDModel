# Material composition and densities from literature
# Rubble generation rate: 0.8 m³/m² built-up area (Tamraz, Srour & Chehab, 2012)

RUBBLE_RATE  = 0.8   # m³ of rubble per m² of built-up area
FLOOR_HEIGHT = 3.0   # metres per floor

STRUCTURES = {
    "Residential Low Rise": {
        "floors": 2,
        "splits": {"concrete": 0.79, "steel": 0.04, "masonry": 0.13, "wood": 0.02, "other": 0.02},
    },
    "Residential High Rise": {
        "floors": 5,
        "splits": {"concrete": 0.79, "steel": 0.04, "masonry": 0.13, "wood": 0.02, "other": 0.02},
    },
    "Industrial": {
        "floors": 1,
        "splits": {"concrete": 0.79, "steel": 0.04, "masonry": 0.13, "wood": 0.02, "other": 0.02},
    },
}

DENSITIES = {
    "concrete": 2400,
    "masonry":  1800,
    "steel":    7850,
    "wood":     600,
    "other":    1500,
}

# ── cleanup constants (Tamraz, Srour & Chehab, 2012) ──────────────────────────
STEEL_ACCESSIBILITY     = 0.70   # fraction of steel tonnage accessible for manual sorting
LABORERS                = 4
LABORER_PRODUCTIVITY    = 0.70   # tonnes/hour per laborer
EXCAVATOR_PRODUCTIVITY  = 70.51  # m³/hour  (CAT 320DL, thumb attachment)
LOADER_PRODUCTIVITY     = 160.9  # m³/hour  (CAT 950GC)
WORKDAY_HOURS           = 8


def quantify_building(building, gsd, structure_type):
    config  = STRUCTURES[structure_type]
    floors  = config["floors"]

    area_m2         = building["pixel_area"] * (gsd ** 2)
    built_up_m2     = area_m2 * floors
    height_m        = floors * FLOOR_HEIGHT
    rubble_volume_m3 = built_up_m2 * RUBBLE_RATE

    masses = {
        mat: rubble_volume_m3 * frac * DENSITIES[mat]
        for mat, frac in config["splits"].items()
    }
    total_kg = sum(masses.values())

    # ── cleanup time ──────────────────────────────────────────────────────────
    steel_t          = masses["steel"] / 1000                               # kg → tonnes
    accessible_steel = steel_t * STEEL_ACCESSIBILITY
    manual_hrs       = accessible_steel / (LABORERS * LABORER_PRODUCTIVITY)

    concrete_vol     = rubble_volume_m3 * config["splits"]["concrete"]
    excavator_hrs    = concrete_vol / EXCAVATOR_PRODUCTIVITY

    loader_hrs       = rubble_volume_m3 / LOADER_PRODUCTIVITY

    total_hrs        = manual_hrs + excavator_hrs + loader_hrs
    total_days       = total_hrs / WORKDAY_HOURS

    return {
        "building_id":       building["id"],
        "pixel_area":        building["pixel_area"],
        "area_m2":           round(area_m2, 2),
        "built_up_m2":       round(built_up_m2, 2),
        "height_m":          round(height_m, 2),
        "rubble_volume_m3":  round(rubble_volume_m3, 2),
        "concrete_kg":       round(masses["concrete"], 2),
        "steel_kg":          round(masses["steel"], 2),
        "masonry_kg":        round(masses["masonry"], 2),
        "wood_kg":           round(masses["wood"], 2),
        "other_kg":          round(masses["other"], 2),
        "mass_kg":           round(total_kg, 2),
        # cleanup
        "manual_sort_hrs":   round(manual_hrs, 2),
        "excavator_hrs":     round(excavator_hrs, 2),
        "loader_hrs":        round(loader_hrs, 2),
        "total_cleanup_hrs": round(total_hrs, 2),
        "total_cleanup_days": round(total_days, 2),
    }