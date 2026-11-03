"""Configuration shared by all experiment scripts."""

from pathlib import Path

import numpy as np

from tomo import digits_phantom, shepp_logan, system_matrix, angles

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"

N = 100                      # reconstruction grid N x N
F = 4                        # data are simulated on an (F N) x (F N) grid
ANGLES = [5, 10, 20, 45, 90, 180]
NOISE = [0.0, 0.01, 0.05]
SEEDS = [0, 1, 2]
LSQR_KMAX = 320
LSQR_GRID = [1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 80, 113, 160, 226, 320]
TIK_LAMBDAS = list(np.round(np.logspace(-2, 0.5, 11), 6))   # multiplied by sqrt(m)
TV_LAMBDAS = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]    # multiplied by m
TV_ITERS = 600
CRIME_ANGLES = [10, 45, 180]
CRIME_NOISE = [0.01, 0.05]
TAU = 1.05                   # safety factor of the discrepancy principle
PHANTOMS = ["digits", "shepp_logan"]
METHODS = ["FBP", "LSQR", "Tikhonov", "TV"]
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def phantoms_fine():
    return {"digits": digits_phantom(N * F), "shepp_logan": shepp_logan(N * F)}


def coarse(img):
    """Ground truth on the reconstruction grid: F x F block average of the fine image."""
    return img.reshape(N, F, N, F).mean(axis=(1, 3))


def make_data(hi, m, nd, span=180.0):
    """Project the fine image onto an F-times finer detector and bin to the coarse detector."""
    Ah, _ = system_matrix(N * F, angles(m, span), n_det=nd * F)
    return (Ah @ hi.ravel()).reshape(m, nd, F).sum(-1).ravel() / F ** 2


def add_noise(b, eta, m, seed):
    rng = np.random.default_rng([m, seed, int(1e4 * eta)])
    e = rng.standard_normal(b.size)
    e = eta * np.linalg.norm(b) * e / np.linalg.norm(e)
    return b + e, e


def plot_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 8, "savefig.bbox": "tight",
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def save(fig, name):
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
