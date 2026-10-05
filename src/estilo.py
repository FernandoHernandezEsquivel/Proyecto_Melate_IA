"""Estilo de gráficas compartido por los notebooks del proyecto."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Un color fijo por juego (la identidad nunca cambia entre gráficas).
COLORES = {"Melate": "#2a78d6", "Revancha": "#eb6834", "Revanchita": "#1baf7a"}
TEORICO = "#52514e"  # referencia teórica / esperado
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
EJE = "#c3c2b7"
SUPERFICIE = "#fcfcfb"

AZUL_SEQ = LinearSegmentedColormap.from_list(
    "azul_seq", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_rojo", ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"]
)


def aplicar():
    mpl.rcParams.update(
        {
            "figure.facecolor": SUPERFICIE,
            "axes.facecolor": SUPERFICIE,
            "savefig.facecolor": SUPERFICIE,
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "font.family": ["Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "text.color": TINTA,
            "axes.labelcolor": TINTA_2,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.edgecolor": EJE,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": TINTA_2,
            "ytick.labelcolor": TINTA_2,
            "ytick.left": False,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "patch.linewidth": 0,
        }
    )
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(COLORES.values()))
