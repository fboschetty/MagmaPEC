import geoplot as gp
import matplotlib.lines as l
import matplotlib.pyplot as plt
import numpy as np
from labellines import labelLine

from MagmaPEC.tools import FeO_Target


def PEC_plot(name, equilibration, correction, FeO_target, PEC_amount, **kwargs):

    gp.layout(colors=gp.colors.bright, facecolor="white", gridcolor="whitesmoke")
    set_markers = kwargs.get("markers", True)
    fontsize = kwargs.get("fontsize", 12)
    linewidth = kwargs.get("linewidth", 5)
    markersize = kwargs.get("markersize", 70)

    colors = gp.colors.bright.by_key()["color"]
    eq_color = colors[6]
    corr_color = colors[5]

    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=False)

    FeO_color = tuple(np.repeat(0.25, 3))

    plt.plot(
        equilibration["MgO"],
        equilibration["FeO"],
        ["-", ".-"][set_markers],
        color=eq_color,
        # label="equilibration",
        linewidth=linewidth,
        mec="k",
        markersize=10,
        # alpha=0.7,
    )
    plt.plot(
        correction["MgO"],
        correction["FeO"],
        ["-", ".-"][set_markers],
        color=corr_color,
        linewidth=linewidth,
        mec="k",
        markersize=10,
        # alpha=0.7,
    )
    ax.scatter(
        equilibration.loc[equilibration.index[0], "MgO"],
        equilibration.loc[equilibration.index[0], "FeO"],
        marker="^",
        color=colors[3],
        edgecolors="k",
        s=markersize,
        zorder=10,
        label="Glass",
    )
    ax.scatter(
        equilibration.loc[equilibration.index[-1], "MgO"],
        equilibration.loc[equilibration.index[-1], "FeO"],
        marker="o",
        edgecolors="k",
        color=eq_color,
        s=markersize,
        zorder=10,
        label="equilibration",
    )
    ax.scatter(
        correction.loc[correction.index[-1], "MgO"],
        correction.loc[correction.index[-1], "FeO"],
        marker="s",
        color=corr_color,
        edgecolors="k",
        s=markersize,
        zorder=10,
        label="correction",
    )

    middle = sum(ax.get_xlim()) / 2

    if isinstance(FeO_target, FeO_Target):
        FeO_inital = FeO_target.target(melt_wtpc=correction)
        ax.plot(
            correction["MgO"],
            FeO_inital,
            "-",
            color=FeO_color,
        )
        FeO_target = sum((min(FeO_inital), max(FeO_inital))) / 2
    else:
        ax.axhline(FeO_target, linestyle="-", color=FeO_color, linewidth=1.5)

    FeO_line = ax.get_lines()[-1]
    try:
        labelLine(
            FeO_line,
            x=middle,
            label="initial FeO",
            size=fontsize * 0.8,
            color=FeO_color,
        )
    except ValueError:
        pass

    ax.set_ylim(ax.get_ylim()[0], max((FeO_target * 1.04, ax.get_ylim()[1])))

    ax.set_xlabel("MgO (wt. %)", size=fontsize)
    ax.set_ylabel("FeO$^T$\n(wt. %)", rotation=0, labelpad=30, size=fontsize)

    handles, labels = ax.get_legend_handles_labels()

    handles = handles + [l.Line2D([0], [0], linewidth=0)]

    labels = labels + [f"{PEC_amount:.2f} % PEC"]

    legend = ax.legend(
        handles,
        labels,
        title=name,
        prop={"size": fontsize * 0.8},
        fancybox=False,
        facecolor="white",
        framealpha=1,
        frameon=True,
    )

    frame = legend.get_frame()
    frame.set_linewidth(0.0)

    plt.show()
