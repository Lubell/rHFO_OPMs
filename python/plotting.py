"""Shared plotting helpers."""

import matplotlib.pyplot as plt

import config as cfg

FRAME_COLORS = {cfg.ERG_UPPER: "red", cfg.ERG_UNDER: "#006400"}
MS_TICKS = [-0.05, -0.025, 0.0, 0.025, 0.05, 0.075, 0.1]


def sensor_grid(draw, channels=range(16), figsize=(16, 11)):
    """One subplot per channel, laid out like the mounted sensor array.

    `draw(ax, ch)` plots channel `ch` into `ax`. ERG panels get a coloured frame.
    """
    fig = plt.figure(figsize=figsize)
    for ch in channels:
        ax = fig.add_subplot(4, 5, cfg.GRID_POS[ch])
        draw(ax, ch)
        if ch in FRAME_COLORS:
            for spine in ax.spines.values():
                spine.set_color(FRAME_COLORS[ch])
        ax.set_title(str(cfg.SENSOR_LABELS[ch]), fontsize=9)
    fig.tight_layout()
    return fig


def ms_xaxis(ax, ticks=MS_TICKS):
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t * 1000:.0f}" for t in ticks])
