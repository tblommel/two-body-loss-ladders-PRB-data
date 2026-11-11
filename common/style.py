"""The single plot style of the paper figures (APS sizes; 9 pt labels, 8 pt ticks; drawn at printed size).
Method identity: color + line style per data label; display labels HF -> MF, TPPX_0 -> TPPX$_0$, mGW -> m$GW$."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402,F401

W1, W2 = 3.4, 7.0                                     # single / double column width [inch]
INK = "#2b2a28"
LW, LWX = 1.15, 1.6
RC = {
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8.5,
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "figure.dpi": 150, "legend.frameon": False,
    "legend.handlelength": 2.2, "legend.columnspacing": 1.0, "legend.handletextpad": 0.5}
plt.rcParams.update(RC)

LABEL = {"exact": "exact", "HF": "MF", "2B": "2B", "TPPX": "TPPX", "TPPX_0": r"TPPX$_0$", "mGW": r"m$GW$",
         "DSL": "DSL"}
STYLE = {"exact": dict(color="#0b0b0b", ls="-"),
         "HF": dict(color="#8a8984", ls=(0, (1, 1.5))),
         "2B": dict(color="#eda100", ls=(0, (6, 2))),
         "TPPX": dict(color="#2a78d6", ls="--"),
         "TPPX_0": dict(color="#2a78d6", ls="-."),
         "mGW": dict(color="#6250d6", ls=(0, (3, 1, 1, 1, 1, 1))),
         "DSL": dict(color="#eb6834", ls="-")}


def style(m, **kw):
    s = dict(STYLE[m]); s["lw"] = LWX if m == "exact" else LW; s["label"] = LABEL[m]
    s.update(kw)
    return s


def legend_handles(methods):
    return [Line2D([], [], **{k: v for k, v in style(m).items() if k in ("color", "ls", "lw")}) for m in methods]


def panel(ax, tag, x=0.0, y=1.03):
    ax.text(x, y, tag, transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", fontsize=9)


TAGPOS = {"ll": (0.04, 0.05, "left", "bottom"), "ul": (0.04, 0.95, "left", "top"),
          "ur": (0.96, 0.95, "right", "top"), "lr": (0.96, 0.05, "right", "bottom")}


def tag(ax, text, loc="ll"):
    """in-panel label (start state, rate) with a white backing"""
    x, y, ha, va = TAGPOS[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8.5, zorder=10,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def save(fig, path_noext):
    fig.savefig(str(path_noext) + ".pdf")
