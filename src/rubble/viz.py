import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider, RadioButtons, CheckButtons

from quantify import RUBBLE_RATE

INIT_THRESHOLD  = 0.5
INIT_STRUCTURE  = "Residential Low Rise"
STRUCTURE_TYPES = ["Residential Low Rise", "Residential High Rise", "Industrial"]
MAX_ROWS        = 15

# ── table column definitions — two modes ──────────────────────────────────────
MASS_COLS   = ["ID", "m²", "Built-up m²", "Rubble m³", "Concrete kg", "Steel kg", "Masonry kg", "Total kg"]
MASS_WIDTHS = [0.05,  0.09,  0.11,          0.10,         0.14,          0.11,       0.13,          0.12]

CLEAN_COLS   = ["ID", "m²", "Rubble m³", "Manual hrs", "Excavator hrs", "Loader hrs", "Total hrs", "Workdays"]
CLEAN_WIDTHS = [0.05,  0.09,  0.10,        0.12,          0.14,            0.12,         0.11,        0.10]


# ── helpers ───────────────────────────────────────────────────────────────────

def _mass_rows(results):
    rows = [
        [r["building_id"], int(r["area_m2"]), int(r["built_up_m2"]),
         int(r["rubble_volume_m3"]), int(r["concrete_kg"]),
         int(r["steel_kg"]), int(r["masonry_kg"]), int(r["mass_kg"])]
        for r in results[:MAX_ROWS]
    ]
    if len(results) > MAX_ROWS:
        rows.append(["...", f"+{len(results)-MAX_ROWS} more", "", "", "", "", "", ""])
    return rows


def _clean_rows(results):
    rows = [
        [r["building_id"], int(r["area_m2"]),
         int(r["rubble_volume_m3"]),
         round(r["manual_sort_hrs"],  1),
         round(r["excavator_hrs"],    1),
         round(r["loader_hrs"],       1),
         round(r["total_cleanup_hrs"],1),
         round(r["total_cleanup_days"],2)]
        for r in results[:MAX_ROWS]
    ]
    if len(results) > MAX_ROWS:
        rows.append(["...", f"+{len(results)-MAX_ROWS} more", "", "", "", "", "", ""])
    return rows


def _draw_ids(ax, buildings):
    for b in buildings:
        x, y = b["centroid"]
        txt = ax.text(x, y, str(b["id"]),
                      color="yellow", fontsize=7, ha="center", va="center", fontweight="bold")
        txt.set_path_effects([pe.Stroke(linewidth=2, foreground="black"), pe.Normal()])


# ── main ──────────────────────────────────────────────────────────────────────

def show(pre, post, truth, probs, gsd, process_fn):

    gsd_warn = gsd > 0.8

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plt.subplots_adjust(left=0.06, right=0.52, bottom=0.25, top=0.93)

    axes[0, 0].imshow(pre);                axes[0, 0].set_title("Pre");           axes[0, 0].axis("off")
    axes[0, 1].imshow(post);               axes[0, 1].set_title("Post");          axes[0, 1].axis("off")
    axes[1, 0].imshow(truth, cmap="gray"); axes[1, 0].set_title("Ground Truth");  axes[1, 0].axis("off")
    axes[1, 1].axis("off");                axes[1, 1].set_title("Prediction")

    gsd_color = "red" if gsd_warn else "dimgray"
    gsd_label = f"GSD: {gsd:.3f} m/px" + \
                ("  ⚠ coarser than target — estimates less reliable" if gsd_warn else "")
    fig.text(0.06, 0.96, gsd_label, color=gsd_color, fontsize=9)

    fig.text(0.55, 0.96,
             f"Rubble rate: {RUBBLE_RATE} m³/m² built-up  (Tamraz, Srour & Chehab, 2012)  |  "
             f"79% concrete · 4% steel · 13% masonry · 4% other",
             color="dimgray", fontsize=8)

    table_ax  = plt.axes([0.55, 0.25, 0.43, 0.60])
    table_ax.axis("off")

    slider_ax = plt.axes([0.10, 0.13, 0.40, 0.03])
    slider    = Slider(slider_ax, "Threshold", 0.0, 1.0, valinit=INIT_THRESHOLD)

    radio_ax  = plt.axes([0.55, 0.03, 0.24, 0.16])
    radio     = RadioButtons(radio_ax, STRUCTURE_TYPES, active=0)

    # cleanup toggle checkbox
    check_ax  = plt.axes([0.82, 0.06, 0.16, 0.08])
    check     = CheckButtons(check_ax, ["Show cleanup"], [False])

    # ── update ────────────────────────────────────────────────────────────────
    def update(_=None):
        threshold      = slider.val
        structure_type = radio.value_selected
        show_cleanup   = check.get_status()[0]

        cleaned, buildings, results = process_fn(threshold, structure_type)

        axes[1, 1].cla()
        axes[1, 1].imshow(cleaned, cmap="gray")
        axes[1, 1].set_title("Prediction")
        axes[1, 1].axis("off")
        _draw_ids(axes[1, 1], buildings)

        table_ax.cla()
        table_ax.axis("off")

        if not results:
            table_ax.text(0.5, 0.5, "No buildings detected.",
                          ha="center", va="center", color="red", fontsize=10, weight="bold")
        else:
            total_rubble = sum(r["rubble_volume_m3"]   for r in results)
            total_mass   = sum(r["mass_kg"]             for r in results)
            total_days   = sum(r["total_cleanup_days"]  for r in results)

            if show_cleanup:
                summary = (f"{len(results)} buildings  |  "
                           f"Total rubble: {total_rubble:,.0f} m³  |  "
                           f"Est. cleanup: {total_days:,.1f} workdays")
                rows   = _clean_rows(results)
                cols   = CLEAN_COLS
                widths = CLEAN_WIDTHS
            else:
                summary = (f"{len(results)} buildings  |  "
                           f"Total rubble: {total_rubble:,.0f} m³  |  "
                           f"Total mass: {total_mass/1000:,.1f} tonnes")
                rows   = _mass_rows(results)
                cols   = MASS_COLS
                widths = MASS_WIDTHS

            table_ax.text(0.0, 0.98, summary,
                          transform=table_ax.transAxes,
                          fontsize=8.5, color="black", weight="bold", va="top")

            t = table_ax.table(cellText=rows, colLabels=cols,
                               colWidths=widths, loc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(7.5)
            t.scale(1, 1.4)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    radio.on_clicked(update)
    check.on_clicked(update)

    update()
    plt.show()