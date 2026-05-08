"""
quant_plot.py — publication-quality figures for PTQ quantization analysis.

Reads:  docs/reports/quant_data.json
Writes: docs/reports/fig1_per_layer_snr.pdf   (+ .png)
        docs/reports/fig2_config_comparison.pdf (+ .png)
        docs/reports/fig3_snr_by_optype.pdf    (+ .png)

Style:  IEEE/Nature journal style (serif, 300 dpi, tight layout)
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

REPORT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "reports"))
DATA_PATH  = os.path.join(REPORT_DIR, "quant_data.json")

# ── Journal rcParams ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "font.size":         8,
    "axes.labelsize":    9,
    "axes.titlesize":    9,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "0.7",
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches":0.05,
    "lines.linewidth":   1.0,
    "axes.linewidth":    0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ── Stage colour palette (colorblind-friendly) ────────────────────
STAGE_COLORS = {
    "Stem":    "#4878CF",   # blue
    "Layer 1": "#6ACC65",   # green
    "Layer 2": "#D65F5F",   # red
    "Layer 3": "#B47CC7",   # purple
    "Layer 4": "#C4AD66",   # gold
    "Head":    "#77BEDB",   # sky
}
STAGE_ORDER = ["Stem", "Layer 1", "Layer 2", "Layer 3", "Layer 4", "Head"]

CONFIG_COLORS  = ["#4878CF", "#E8925A", "#D65F5F", "#6ACC65", "#B47CC7"]
CONFIG_HATCHES = ["",       "--",      "//",      "..",      "xx"]
CONFIG_MARKERS = ["o",      "s",       "^",       "D",       "v"]

# Tag of the entropy (outlier) config
_ENTROPY_TAG = "INT8-PC-Entropy"
_ZOOM_TAGS   = [t for t in ["INT8-PC-MinMax", "INT8-PC-MinMax-HW",
                             "UINT8-PC-MinMax", "INT8-PT-MinMax"]]

# ── helpers ───────────────────────────────────────────────────────
def save(fig, stem):
    for ext in ("pdf", "png"):
        path = os.path.join(REPORT_DIR, f"{stem}.{ext}")
        fig.savefig(path)
        print(f"  saved → {path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════
# Figure 1 — Per-layer SNR + cosine (default config: INT8-PC-MinMax)
# ════════════════════════════════════════════════════════════════
def fig1_per_layer(data, ref_tag="INT8-PC-MinMax"):
    layers = data["per_layer"][ref_tag]
    # Filter to layers that have a quantization scale (skip pure pass-through)
    layers_q = [l for l in layers if l["snr"] is not None]

    ops      = [l["op"] for l in layers_q]
    snrs     = [l["snr"] for l in layers_q]
    cosines  = [l["cosine"] for l in layers_q]
    stages   = [l["stage"] for l in layers_q]
    xs       = np.arange(len(layers_q))

    # Short x-labels: index only (too many to label by name)
    colors = [STAGE_COLORS[s] for s in stages]

    fig = plt.figure(figsize=(7.0, 4.2))
    gs  = GridSpec(2, 1, figure=fig, hspace=0.08, height_ratios=[3, 1.4])

    # ── Panel A: SNR bar chart ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    bars = ax1.bar(xs, snrs, color=colors, width=0.72, zorder=3, linewidth=0)

    ax1.axhline(40, color="0.4", lw=0.8, ls="--", zorder=2, label="40 dB threshold")
    ax1.axhline(30, color="0.65", lw=0.7, ls=":",  zorder=2, label="30 dB threshold")
    ax1.set_xlim(-0.7, len(xs) - 0.3)
    ax1.set_ylim(0, max(snrs) * 1.12)
    ax1.set_ylabel("SNR (dB)", labelpad=4)
    ax1.set_xticklabels([])
    ax1.tick_params(axis="x", bottom=False)
    ax1.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title(
        r"Per-Layer Quantization Quality — ResNet-18 INT8$\times$INT8 "
        r"(per-channel, min-max calibration)",
        pad=5,
    )

    # legend for stages
    patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s) for s in STAGE_ORDER
               if any(st == s for st in stages)]
    l1 = ax1.legend(handles=patches, title="Stage", loc="lower right",
                    ncol=3, framealpha=0.88, fontsize=6.5, title_fontsize=7)
    ax1.add_artist(l1)
    ax1.legend(loc="upper left", fontsize=6.5, handlelength=1.6)

    # annotate min SNR
    min_idx = int(np.argmin(snrs))
    ax1.annotate(
        f"min {snrs[min_idx]:.1f} dB",
        xy=(min_idx, snrs[min_idx]),
        xytext=(min_idx + 2, snrs[min_idx] - 3.5),
        fontsize=6.5,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="0.3"),
        color="0.2",
    )

    # op-type markers on x-axis: small colored triangles
    op_marker = {"Conv": "^", "Gemm": "s", "Add": "D",
                 "MaxPool": "v", "GlobalAveragePool": "o", "Flatten": "x", "Relu": "."}
    for xi, op in zip(xs, ops):
        ax1.plot(xi, -1.5, marker=op_marker.get(op, "."), ms=3.5,
                 color="0.35", clip_on=False, zorder=4)

    # ── Panel B: cosine similarity ──────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    cos_arr = np.array(cosines)
    ax2.plot(xs, (1 - cos_arr) * 1e4, color="#4878CF", lw=0.9, zorder=3)
    ax2.fill_between(xs, 0, (1 - cos_arr) * 1e4, alpha=0.18, color="#4878CF", zorder=2)
    ax2.set_xlim(-0.7, len(xs) - 0.3)
    ax2.set_ylabel(r"$(1 - \cos)\times 10^4$", labelpad=4)
    ax2.set_xlabel("Layer index", labelpad=3)
    ax2.yaxis.grid(True, lw=0.4, alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.set_xticks(xs[::4])

    fig.align_ylabels([ax1, ax2])
    fig.text(0.01, 0.97, "(a)", fontsize=9, fontweight="bold", va="top")
    fig.text(0.01, 0.32, "(b)", fontsize=9, fontweight="bold", va="top")

    save(fig, "fig1_per_layer_snr")


# ════════════════════════════════════════════════════════════════
# Figure 2 — Configuration comparison: two panels
#   (a) all 5 configs including entropy outlier
#   (b) zoomed view of the 4 non-entropy configs
# ════════════════════════════════════════════════════════════════
def fig2_config_comparison(data):
    summary  = data["summary"]
    tag_order = [r["tag"] for r in summary]
    color_map = {r["tag"]: CONFIG_COLORS[i]  for i, r in enumerate(summary)}
    hatch_map = {r["tag"]: CONFIG_HATCHES[i] for i, r in enumerate(summary)}

    metrics = [
        ("snr_first_conv", "1st Conv"),
        ("snr_e2e",        "E2E Out"),
        ("snr_first_add",  "Res-Add"),
        ("snr_fc",         "FC"),
    ]
    m_keys   = [m[0] for m in metrics]
    m_labels = [m[1] for m in metrics]
    n_metric = len(metrics)
    group_xs = np.arange(n_metric)

    zoom_rows = [r for r in summary if r["tag"] in _ZOOM_TAGS]
    all_rows  = summary

    def _draw_bars(ax, rows, bar_w, annotate=True):
        n = len(rows)
        for ci, row in enumerate(rows):
            tag = row["tag"]
            offsets = (ci - (n - 1) / 2) * bar_w
            vals = [row.get(k) for k in m_keys]
            rects = ax.bar(group_xs + offsets, vals,
                           width=bar_w,
                           color=color_map[tag],
                           hatch=hatch_map[tag],
                           label=row["label"].replace("\n", " "),
                           edgecolor="white", linewidth=0.4, zorder=3)
            if annotate:
                for rect, v in zip(rects, vals):
                    if v is not None and rect.get_height() > 0:
                        ax.text(rect.get_x() + rect.get_width() / 2,
                                rect.get_height() + 0.15,
                                f"{v:.1f}",
                                ha="center", va="bottom",
                                fontsize=4.5, color="0.15", rotation=90,
                                clip_on=True)

    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(8.0, 3.2),
                                           gridspec_kw={"wspace": 0.32})

    # ── Panel A: all 5 configs ──────────────────────────────────
    _draw_bars(ax_all, all_rows, bar_w=0.15, annotate=False)
    all_vals = [r.get(k) for r in all_rows for k in m_keys if r.get(k) is not None]
    ax_all.set_ylim(min(all_vals) * 0.90, max(all_vals) * 1.12)
    ax_all.set_xticks(group_xs)
    ax_all.set_xticklabels(m_labels, fontsize=7)
    ax_all.set_ylabel("SNR (dB)", labelpad=4)
    ax_all.set_title("All configurations (incl. entropy outlier)", pad=4, fontsize=8)
    ax_all.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax_all.set_axisbelow(True)
    ax_all.legend(loc="lower right", ncol=1, fontsize=5.5,
                  title="Config", title_fontsize=6.0)
    ax_all.text(0.02, 0.98, "(a)", transform=ax_all.transAxes,
                fontsize=9, fontweight="bold", va="top")

    # ── Panel B: zoom — 4 non-entropy configs ──────────────────
    _draw_bars(ax_zoom, zoom_rows, bar_w=0.18, annotate=True)
    zoom_vals = [r.get(k) for r in zoom_rows for k in m_keys if r.get(k) is not None]
    y_lo = min(zoom_vals)
    y_hi = max(zoom_vals)
    margin = (y_hi - y_lo) * 0.35 or 1.0
    ax_zoom.set_ylim(y_lo - margin, y_hi + margin * 1.8)
    ax_zoom.set_xticks(group_xs)
    ax_zoom.set_xticklabels(m_labels, fontsize=7)
    ax_zoom.set_ylabel("SNR (dB)", labelpad=4)
    ax_zoom.set_title(r"Zoomed: 4 comparable configs $\,$(entropy excluded)",
                      pad=4, fontsize=8)
    ax_zoom.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax_zoom.set_axisbelow(True)
    ax_zoom.legend(loc="lower right", ncol=1, fontsize=5.5,
                   title="Config", title_fontsize=6.0)
    ax_zoom.text(0.02, 0.98, "(b)", transform=ax_zoom.transAxes,
                 fontsize=9, fontweight="bold", va="top")

    fig.suptitle("Quantization Configuration Comparison — ResNet-18",
                 fontsize=9, fontweight="bold", y=1.01)
    save(fig, "fig2_config_comparison")


# ════════════════════════════════════════════════════════════════
# Figure 3 — SNR distribution by operator type (box plot)
# ════════════════════════════════════════════════════════════════
def fig3_snr_by_optype(data, ref_tag="INT8-PC-MinMax"):
    layers = data["per_layer"][ref_tag]
    layers_q = [l for l in layers if l["snr"] is not None]

    op_order = ["Conv", "Add", "Relu", "MaxPool", "GlobalAveragePool", "Gemm"]
    op_labels = ["Conv", "Add\n(Residual)", "ReLU", "MaxPool", "GAP", "Gemm\n(FC)"]

    op_snrs = {}
    for l in layers_q:
        op = l["op"]
        if op not in op_snrs:
            op_snrs[op] = []
        op_snrs[op].append(l["snr"])

    # Include only ops present in data
    present = [(op, lbl) for op, lbl in zip(op_order, op_labels) if op in op_snrs]
    ops_p, labels_p = zip(*present)
    box_data = [op_snrs[op] for op in ops_p]

    # Colour by stage approximation — use op colour
    op_colors = {
        "Conv":              "#4878CF",
        "Add":               "#D65F5F",
        "Relu":              "#6ACC65",
        "MaxPool":           "#B47CC7",
        "GlobalAveragePool": "#77BEDB",
        "Gemm":              "#C4AD66",
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    xs = np.arange(1, len(ops_p) + 1)

    bp = ax.boxplot(box_data, positions=xs, widths=0.5, patch_artist=True,
                    medianprops=dict(color="0.2", lw=1.2),
                    whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8),
                    flierprops=dict(marker="o", ms=3, markeredgewidth=0.5,
                                   markerfacecolor="none"))
    for patch, op in zip(bp["boxes"], ops_p):
        patch.set_facecolor(op_colors.get(op, "0.7"))
        patch.set_alpha(0.75)
        patch.set_linewidth(0.7)

    # Overlay individual points (jitter)
    rng = np.random.default_rng(0)
    for xi, vals in zip(xs, box_data):
        jitter = rng.uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, s=14, color="0.25", alpha=0.7,
                   zorder=5, linewidths=0)

    ax.axhline(40, color="0.4", lw=0.8, ls="--", label="40 dB", zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels_p, fontsize=7)
    ax.set_ylabel("SNR (dB)", labelpad=4)
    ax.set_title("Per-Operator-Type SNR Distribution", pad=5)
    ax.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, handlelength=1.5)

    save(fig, "fig3_snr_by_optype")


# ════════════════════════════════════════════════════════════════
# Figure 4 — Calibration method comparison (all configs, per-layer)
# ════════════════════════════════════════════════════════════════
def fig4_calib_comparison(data):
    """Compare per-layer SNR: minmax vs entropy (int8 per-channel)."""
    tags_to_plot = ["INT8-PC-MinMax", "INT8-PC-Entropy"]
    labels_map   = {r["tag"]: r["label"].replace("\n", " ")
                    for r in data["summary"]}
    lcolors      = ["#4878CF", "#D65F5F"]
    lmarkers     = ["o", "s"]

    all_layers = data["per_layer"]["INT8-PC-MinMax"]
    layer_q = [l for l in all_layers if l["snr"] is not None]
    xs = np.arange(len(layer_q))

    fig, ax = plt.subplots(figsize=(6.0, 2.8))

    for ci, tag in enumerate(tags_to_plot):
        if tag not in data["per_layer"]:
            continue
        snrs = [l["snr"] for l in data["per_layer"][tag] if l["snr"] is not None]
        if len(snrs) != len(xs):
            snrs = snrs[:len(xs)]
        ax.plot(xs, snrs, lw=0.9, color=lcolors[ci],
                marker=lmarkers[ci], ms=2.5, markevery=4,
                label=labels_map.get(tag, tag), zorder=3)

    # Difference fill
    snrs_mm = [l["snr"] for l in data["per_layer"]["INT8-PC-MinMax"]
               if l["snr"] is not None]
    snrs_en = [l["snr"] for l in data["per_layer"]["INT8-PC-Entropy"]
               if l["snr"] is not None]
    n = min(len(snrs_mm), len(snrs_en))
    ax.fill_between(xs[:n], snrs_mm[:n], snrs_en[:n],
                    alpha=0.12, color="#B47CC7",
                    label="Difference region")

    ax.axhline(40, color="0.45", lw=0.7, ls="--", zorder=2, label="40 dB")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_xlabel("Layer index", labelpad=3)
    ax.set_ylabel("SNR (dB)", labelpad=4)
    ax.set_title("Calibration Method Comparison: Min-Max vs. Entropy", pad=5)
    ax.yaxis.grid(True, lw=0.4, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")

    save(fig, "fig4_calib_comparison")


# ════════════════════════════════════════════════════════════════
# Figure 5 — Per-layer absolute & relative error (default config)
# ════════════════════════════════════════════════════════════════
def fig5_error_metrics(data, ref_tag="INT8-PC-MinMax"):
    layers_q = [l for l in data["per_layer"][ref_tag] if l.get("mae") is not None]
    xs      = np.arange(len(layers_q))
    maes    = np.array([l["mae"]          for l in layers_q])
    rmses   = np.array([l["rmse"]         for l in layers_q])
    max_aes = np.array([l["max_ae"]       for l in layers_q])
    mres    = np.array([l["mean_re_pct"]  for l in layers_q])
    nrmses  = np.array([l["nrmse_pct"]    for l in layers_q])
    stages  = [l["stage"] for l in layers_q]
    ops     = [l["op"]    for l in layers_q]
    colors  = [STAGE_COLORS[s] for s in stages]

    fig = plt.figure(figsize=(7.0, 5.0))
    gs  = GridSpec(3, 1, figure=fig, hspace=0.10, height_ratios=[2.5, 2, 1.5])

    # ── Panel A: MAE + Max-AE ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.bar(xs, max_aes, color=[c + "55" for c in [STAGE_COLORS[s] for s in stages]],
            width=0.75, label="Max |error|", zorder=2, linewidth=0)
    ax1.bar(xs, maes, color=colors, width=0.75,
            label="MAE", zorder=3, linewidth=0)
    ax1.set_xlim(-0.7, len(xs) - 0.3)
    ax1.set_ylabel("Absolute Error\n(float32 units)", labelpad=4)
    ax1.set_xticklabels([])
    ax1.tick_params(axis="x", bottom=False)
    ax1.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title(
        r"Per-Layer Floating-Point Error — ResNet-18 INT8$\times$INT8 "
        r"(per-channel, min-max)", pad=5)
    ax1.legend(fontsize=6.5, loc="upper left", ncol=2)

    # stage legend
    patches = [mpatches.Patch(color=STAGE_COLORS[s], label=s)
               for s in STAGE_ORDER if any(st == s for st in stages)]
    ax1.legend(handles=patches + [
        mpatches.Patch(color="0.75", label="Max |error|"),
        mpatches.Patch(color="0.35", label="MAE"),
    ], fontsize=6.0, loc="upper right", ncol=4, title="Stage / Metric",
               title_fontsize=6.5)

    # ── Panel B: Mean & Normalised relative error (%) ──────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.bar(xs, mres, color=colors, width=0.75, label="Mean RE (%)", zorder=3, linewidth=0)
    ax2r = ax2.twinx()
    ax2r.plot(xs, nrmses, color="#C4AD66", lw=0.9, zorder=4,
              marker=".", ms=2.0, markevery=3, label="NRMSE (%)")
    ax2r.set_ylabel("NRMSE (%)", labelpad=4, color="#C4AD66")
    ax2r.tick_params(axis="y", labelcolor="#C4AD66", labelsize=6.5)
    ax2r.spines["right"].set_visible(True)
    ax2r.spines["right"].set_linewidth(0.6)
    ax2.set_ylabel("Mean Rel. Error (%)", labelpad=4)
    ax2.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_xticklabels([])
    ax2.tick_params(axis="x", bottom=False)
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2r, labels2r = ax2r.get_legend_handles_labels()
    ax2.legend(lines2 + lines2r, labels2 + labels2r,
               fontsize=6.5, loc="upper left", ncol=2)

    # ── Panel C: RMSE (same units as MAE) ──────────────────────
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.fill_between(xs, rmses, alpha=0.55, color="#4878CF", step="mid", zorder=2)
    ax3.step(xs, rmses, color="#4878CF", lw=0.9, where="mid", zorder=3, label="RMSE")
    ax3.set_xlim(-0.7, len(xs) - 0.3)
    ax3.set_ylabel("RMSE\n(float32 units)", labelpad=4)
    ax3.set_xlabel("Layer index", labelpad=3)
    ax3.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax3.set_axisbelow(True)
    ax3.set_xticks(xs[::4])
    ax3.legend(fontsize=6.5, loc="upper left")

    fig.align_ylabels([ax1, ax2, ax3])
    for label, ax_ref in [("(a)", ax1), ("(b)", ax2), ("(c)", ax3)]:
        y = ax_ref.get_position().y1 + 0.005
        fig.text(0.01, y, label, fontsize=9, fontweight="bold", va="bottom")

    save(fig, "fig5_error_metrics")


# ════════════════════════════════════════════════════════════════
# Figure 6 — MAE comparison: outlier separation + zoom
#
# Layout  (3 columns):
#   Left col  (wide): per-layer MAE, log scale, all 5 configs
#   Mid  col  (wide): per-layer MAE, linear zoom, 4 non-entropy configs
#   Right col (narrow): mean-MAE summary bar chart (all 5)
# ════════════════════════════════════════════════════════════════
def fig6_config_error(data):
    summary    = data["summary"]
    tag_order  = [r["tag"] for r in summary]
    labels_map = {r["tag"]: r["label"].replace("\n", " ") for r in summary}
    color_map  = {r["tag"]: CONFIG_COLORS[i]  for i, r in enumerate(summary)}
    marker_map = {r["tag"]: CONFIG_MARKERS[i] for i, r in enumerate(summary)}
    hatch_map  = {r["tag"]: CONFIG_HATCHES[i] for i, r in enumerate(summary)}

    # Load per-layer data
    config_data = {}
    for tag in tag_order:
        lq = [l for l in data["per_layer"][tag] if l.get("mae") is not None]
        config_data[tag] = {
            "mae":  np.array([l["mae"]         for l in lq]),
            "mre":  np.array([l["mean_re_pct"] for l in lq]),
        }
    xs = np.arange(len(list(config_data.values())[0]["mae"]))

    zoom_tags = [t for t in tag_order if t in _ZOOM_TAGS]

    # Compute thresholds for zoom y-axis
    zoom_mae_all = np.concatenate([config_data[t]["mae"] for t in zoom_tags])
    zoom_y_hi = float(np.percentile(zoom_mae_all, 98)) * 1.25

    # ── Figure layout ──────────────────────────────────────────
    fig = plt.figure(figsize=(10.0, 3.8))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.30,
                   width_ratios=[2.6, 2.6, 1.0])
    ax_log  = fig.add_subplot(gs[0])
    ax_zoom = fig.add_subplot(gs[1])
    ax_bar  = fig.add_subplot(gs[2])

    def _plot_lines(ax, tags):
        for tag in tags:
            cd = config_data[tag]
            ax.plot(xs, cd["mae"],
                    lw=0.85, color=color_map[tag],
                    marker=marker_map[tag], ms=1.8, markevery=6,
                    label=labels_map[tag], zorder=3)

    # ── Left: log-scale, all 5 ─────────────────────────────────
    _plot_lines(ax_log, tag_order)
    ax_log.set_yscale("log")
    ax_log.set_ylabel("MAE (float32 units, log)", labelpad=4)
    ax_log.set_xlabel("Layer index", labelpad=3)
    ax_log.set_title("All configurations (log scale)", pad=4, fontsize=8)
    ax_log.yaxis.grid(True, lw=0.4, alpha=0.4, which="both", zorder=0)
    ax_log.set_axisbelow(True)
    ax_log.set_xlim(-0.5, len(xs) - 0.5)
    ax_log.set_xticks(xs[::6])
    # shade the entropy spikes
    ax_log.axhspan(0.1, ax_log.get_ylim()[1] if ax_log.get_ylim()[1] > 0.1 else 1.5,
                   alpha=0.06, color="#D65F5F", zorder=0)
    ax_log.legend(fontsize=5.5, ncol=1, loc="upper left",
                  title="Config", title_fontsize=6.0)
    ax_log.text(0.02, 0.98, "(a)", transform=ax_log.transAxes,
                fontsize=9, fontweight="bold", va="top")

    # ── Middle: linear zoom, 4 non-entropy ────────────────────
    _plot_lines(ax_zoom, zoom_tags)
    ax_zoom.set_ylim(0, zoom_y_hi)
    ax_zoom.set_ylabel("MAE (float32 units)", labelpad=4)
    ax_zoom.set_xlabel("Layer index", labelpad=3)
    ax_zoom.set_title(r"Zoomed — 4 comparable configs", pad=4, fontsize=8)
    ax_zoom.yaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax_zoom.set_axisbelow(True)
    ax_zoom.set_xlim(-0.5, len(xs) - 0.5)
    ax_zoom.set_xticks(xs[::6])
    # draw a zoom-bracket annotation
    ax_zoom.annotate("", xy=(0.98, 0.98), xycoords="axes fraction",
                     xytext=(0.98, 0.02),
                     arrowprops=dict(arrowstyle="<->", color="0.5", lw=0.8))
    ax_zoom.text(0.995, 0.50, f"0 – {zoom_y_hi:.3f}", transform=ax_zoom.transAxes,
                 fontsize=5.5, color="0.4", va="center", ha="left", rotation=90)
    ax_zoom.legend(fontsize=5.5, ncol=1, loc="upper right",
                   title="Config", title_fontsize=6.0)
    ax_zoom.text(0.02, 0.98, "(b)", transform=ax_zoom.transAxes,
                 fontsize=9, fontweight="bold", va="top")

    # ── Right: mean-MAE summary bar ───────────────────────────
    mean_maes = {tag: float(config_data[tag]["mae"].mean()) for tag in tag_order}
    bar_ys    = np.arange(len(tag_order))
    bar_vals  = [mean_maes[t] for t in tag_order]
    bar_cols  = [color_map[t] for t in tag_order]
    bar_htch  = [hatch_map[t] for t in tag_order]
    bars = ax_bar.barh(bar_ys, bar_vals, color=bar_cols, hatch=bar_htch,
                       edgecolor="white", linewidth=0.4, zorder=3)
    for yi, v in zip(bar_ys, bar_vals):
        ax_bar.text(v + max(bar_vals) * 0.03, yi, f"{v:.4f}",
                    va="center", ha="left", fontsize=5.0, color="0.2")
    ax_bar.set_yticks(bar_ys)
    short_labels = [t.replace("INT8-PC-","").replace("UINT8-PC-","U")
                      .replace("INT8-PT-","PT-") for t in tag_order]
    ax_bar.set_yticklabels(short_labels, fontsize=5.5)
    ax_bar.set_xlabel("Mean MAE", labelpad=3, fontsize=7)
    ax_bar.set_title("Mean MAE\nsummary", pad=4, fontsize=7)
    ax_bar.xaxis.grid(True, lw=0.4, alpha=0.5, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.invert_yaxis()
    ax_bar.text(0.05, 0.98, "(c)", transform=ax_bar.transAxes,
                fontsize=9, fontweight="bold", va="top")

    fig.suptitle(
        "Per-Layer MAE: Outlier Separation and Configuration Zoom — ResNet-18",
        fontsize=9, fontweight="bold", y=1.02)
    save(fig, "fig6_config_error")


# ── main ──────────────────────────────────────────────────────────
def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found: {DATA_PATH}")
        print("Run tools/quant_export_data.py first.")
        sys.exit(1)

    with open(DATA_PATH) as f:
        data = json.load(f)

    print("Generating figures...")
    fig1_per_layer(data)
    fig2_config_comparison(data)
    fig3_snr_by_optype(data)
    fig4_calib_comparison(data)
    fig5_error_metrics(data)
    fig6_config_error(data)
    print("\nAll figures saved to", REPORT_DIR)

if __name__ == "__main__":
    main()
