import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Slider, RadioButtons

from quantify import COLLAPSE_FACTOR


INIT_THRESHOLD  = 0.5
INIT_STRUCTURE  = "Residential Low Rise"
STRUCTURE_TYPES = ["Residential Low Rise", "Residential High Rise", "Industrial"]

MAX_ROWS = 15
COLS     = ["ID", "m²", f"m³ (×{COLLAPSE_FACTOR})", "Concrete kg", "Steel kg", "Masonry kg", "Total kg"]
WIDTHS   = [0.06, 0.10, 0.13,                        0.17,          0.13,       0.13,         0.13]


# ── helpers ───────────────────────────────────────────────────────────────────

def _table_rows(results):
    rows = [
        [r["building_id"], int(r["area_m2"]), int(r["rubble_volume_m3"]),
         int(r["concrete_kg"]), int(r["steel_kg"]), int(r["masonry_kg"]), int(r["mass_kg"])]
        for r in results[:MAX_ROWS]
    ]
    if len(results) > MAX_ROWS:
        rows.append(["...", f"+{len(results) - MAX_ROWS} more", "", "", "", "", ""])
    return rows


def _draw_ids(ax, buildings):
    """Stamp each building's ID number at its centroid on the prediction panel."""
    for b in buildings:
        x, y = b["centroid"]
        txt = ax.text(x, y, str(b["id"]),
                      color="yellow", fontsize=7, ha="center", va="center", fontweight="bold")
        txt.set_path_effects([pe.Stroke(linewidth=2, foreground="black"), pe.Normal()])


# ── main ──────────────────────────────────────────────────────────────────────

def show(pre, post, truth, probs, gsd, process_fn):

    gsd_warn = gsd > 0.8

    # ── layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(left=0.06, right=0.52, bottom=0.22, top=0.93)

    axes[0, 0].imshow(pre);               axes[0, 0].set_title("Pre");            axes[0, 0].axis("off")
    axes[0, 1].imshow(post);              axes[0, 1].set_title("Post");           axes[0, 1].axis("off")
    axes[1, 0].imshow(truth, cmap="gray"); axes[1, 0].set_title("Ground Truth");  axes[1, 0].axis("off")
    axes[1, 1].axis("off");               axes[1, 1].set_title("Prediction")

    # GSD info line above the plots
    gsd_color = "red" if gsd_warn else "dimgray"
    gsd_label = f"GSD: {gsd:.3f} m/px (pan_resolution)" + ("  ⚠ coarser than xBD target — area estimates less reliable" if gsd_warn else "")
    fig.text(0.06, 0.96, gsd_label, color=gsd_color, fontsize=9)

    # Collapse factor annotation above the table
    fig.text(0.55, 0.96,
             f"Collapse factor: {COLLAPSE_FACTOR:.0%}  —  mass estimates assume {COLLAPSE_FACTOR:.0%} of building volume becomes rubble",
             color="dimgray", fontsize=9)

    table_ax = plt.axes([0.55, 0.22, 0.43, 0.62])
    table_ax.axis("off")

    slider_ax = plt.axes([0.10, 0.10, 0.40, 0.03])
    slider    = Slider(slider_ax, "Threshold", 0.0, 1.0, valinit=INIT_THRESHOLD)

    radio_ax = plt.axes([0.55, 0.03, 0.22, 0.14])
    radio    = RadioButtons(radio_ax, STRUCTURE_TYPES, active=0)

    # ── update ────────────────────────────────────────────────────────────────
    def update(_=None):
        threshold      = slider.val
        structure_type = radio.value_selected

        cleaned, buildings, results = process_fn(threshold, structure_type)

        # prediction panel
        axes[1, 1].cla()
        axes[1, 1].imshow(cleaned, cmap="gray")
        axes[1, 1].set_title("Prediction")
        axes[1, 1].axis("off")
        _draw_ids(axes[1, 1], buildings)

        # table
        table_ax.cla()
        table_ax.axis("off")

        if not results:
            table_ax.text(0.5, 0.5, "No buildings detected.", ha="center", va="center",
                          color="red", fontsize=10, weight="bold")
        else:
            t = table_ax.table(cellText=_table_rows(results), colLabels=COLS,
                               colWidths=WIDTHS, loc="center")
            t.auto_set_font_size(False)
            t.set_fontsize(7.5)
            t.scale(1, 1.4)

        fig.canvas.draw_idle()

    slider.on_changed(update)
    radio.on_clicked(update)

    update()
    plt.show()