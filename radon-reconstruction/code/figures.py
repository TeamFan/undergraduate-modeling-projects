"""Draw every figure of the report from the files in results/.

    python code/figures.py"""

import numpy as np
import pandas as pd

from common import ANGLES, CB, METHODS, N, NOISE, RES, plot_style, save
from tomo import metrics

plt = plot_style()
LABEL = {"digits": "digits", "shepp_logan": "Shepp--Logan".replace("--", "–")}


def fig_spectrum():
    sv = pd.read_csv(RES / "singular_values.csv")
    pic = pd.read_csv(RES / "picard.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for t, m in enumerate([10, 45, 180]):
        s = sv[sv.angles == m].sigma.values
        s = s[s > 1e-7 * s[0]]
        ax.semilogy(np.arange(1, s.size + 1), s / s[0], ["-", "--", ":"][t], c=CB[t], label=f"{m} angles")
    ax.axvline(N * N, c="#999999", lw=0.8)
    ax.set_xlabel("index $i$")
    ax.set_ylabel(r"$\sigma_i/\sigma_1$")
    ax.set_title("a) singular values", loc="left", fontsize=8)
    ax.legend(frameon=False)
    ax = axes[1]
    p = pic[pic.phantom == "shepp_logan"].sort_values("i")

    def smooth(v, w=51):
        lv = np.log(np.maximum(v, 1e-300))
        return np.exp(np.convolve(lv, np.ones(w) / w, mode="same"))

    i = p.i.values
    ax.semilogy(i, p.sigma.values, c="k", lw=1.0, label=r"$\sigma_i$")
    for t, (col, lab) in enumerate([("coef_consistent", r"$|u_i^\top A x_{\rm true}|$"),
                                     ("coef_fine", r"$|u_i^\top b|$, noise-free"),
                                     ("coef_noise5", r"$|u_i^\top b|$, 5% noise")]):
        ax.semilogy(i, p[col].values, ".", c=CB[t], ms=0.6, alpha=0.25, rasterized=True)
        ax.semilogy(i[25:-25], smooth(p[col].values)[25:-25], c=CB[t], lw=1.2, label=lab)
    ax.set_xlabel("index $i$")
    ax.set_title("b) Picard plot, Shepp–Logan, 180 angles", loc="left", fontsize=8)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    save(fig, "spectrum")


def fig_nullspace():
    d = np.load(RES / "nullspace_digits_10.npz")
    bnd = pd.read_csv(RES / "nullspace_bound.csv")
    nf = float(bnd[(bnd.phantom == "digits") & (bnd.angles == 10)].null_fraction.iloc[0])
    fig, axes = plt.subplots(1, 3, figsize=(5.6, 2.0))
    lim = np.abs(d["null_part"]).max()
    for ax, img, lab, kw in zip(axes, [d["truth"], d["range_part"], d["null_part"]],
                                ["a) $x_{\\rm true}$", "b) $P_{\\mathcal{R}(A^\\top)}x_{\\rm true}$",
                                 f"c) $P_{{\\mathcal{{N}}(A)}}x_{{\\rm true}}$ ({100 * nf:.0f}% of the norm)"],
                                [dict(cmap="gray", vmin=0, vmax=1), dict(cmap="gray", vmin=0, vmax=1),
                                 dict(cmap="RdBu_r", vmin=-lim, vmax=lim)]):
        ax.imshow(img, **kw)
        ax.set_title(lab, fontsize=7, loc="left")
        ax.axis("off")
    save(fig, "nullspace")


def fig_error_vs_angles():
    best = pd.read_csv(RES / "best_by_method.csv")
    mk = {"FBP": "o-", "LSQR": "s--", "Tikhonov": "^-.", "TV": "d:"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, sharey="row")
    for r, ph in enumerate(["digits", "shepp_logan"]):
        for c, eta in enumerate(NOISE):
            ax = axes[r, c]
            for t, meth in enumerate(METHODS):
                s = best[(best.phantom == ph) & (best.noise == eta) & (best.method == meth)].sort_values("angles")
                ax.errorbar(s.angles, s.rel_err, yerr=s.rel_err_std.fillna(0), fmt=mk[meth], c=CB[t], ms=4,
                            capsize=2, label=meth)
            ax.axvline(N * N / 143, c="#cccccc", lw=0.8)
            ax.set_xscale("log")
            ax.set_xlim(4, 220)
            ax.set_xticks(ANGLES)
            ax.set_xticklabels(ANGLES)
            ax.minorticks_off()
            ax.set_title(f"{chr(97 + 3 * r + c)}) {LABEL[ph]}, {100 * eta:.0f}% noise", loc="left", fontsize=8)
            if r == 1:
                ax.set_xlabel("number of angles $m$")
            if c == 0:
                ax.set_ylabel("relative error")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03), frameon=False)
    save(fig, "error_vs_angles")


def fig_reconstructions():
    ex = np.load(RES / "recon_examples.npz")
    cases = [("digits", 10), ("digits", 180), ("shepp_logan", 10), ("shepp_logan", 180)]
    fig, axes = plt.subplots(len(cases), 5, figsize=(7.2, 6.4))
    for r, (ph, m) in enumerate(cases):
        truth = ex[f"{ph}_truth"]
        imgs = [("ground truth", truth)] + [(k, ex[f"{ph}_{m}_1_{k}"]) for k in METHODS]
        for c, (lab, img) in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
            if lab != "ground truth":
                e = metrics(img, truth)
                lab = f"{lab}: {e['rel_err']:.2f} / {e['ssim']:.2f}"
            else:
                lab = f"{LABEL[ph]}, {m} angles"
            ax.set_title(lab, fontsize=7)
            ax.axis("off")
    save(fig, "reconstructions")


def fig_parameters():
    paths = pd.read_csv(RES / "lsqr_paths.csv")
    raw = pd.read_csv(RES / "all_runs.csv")
    raw = raw[(raw["mode"] == "fine") & (raw.phantom == "digits")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    styles = {0.01: "-", 0.05: "--"}
    for t, m in enumerate([10, 45, 180]):
        for eta, ls in styles.items():
            g = paths[(paths.phantom == "digits") & (paths.angles == m) & (paths.noise == eta)]
            g = g.groupby("k").rel_err.mean()
            axes[0].semilogx(g.index, g.values, ls, c=CB[t], label=f"{m} angles" if eta == 0.01 else None)
            for ax, meth, scale in ((axes[1], "Tikhonov", np.sqrt(m)), (axes[2], "TV", m)):
                h = raw[(raw.angles == m) & (raw.noise == eta) & (raw.method == meth)]
                h = h.groupby("param").rel_err.mean()
                ax.semilogx(h.index / scale, h.values, ls + ("o" if meth == "TV" else ""), c=CB[t], ms=3)
    axes[0].set_xlabel("iteration $k$")
    axes[1].set_xlabel(r"$\lambda/\sqrt{m}$")
    axes[2].set_xlabel(r"$\lambda/m$")
    axes[0].set_ylabel("relative error")
    for ax, ttl in zip(axes, ["a) LSQR", "b) Tikhonov", "c) TV"]):
        ax.set_title(ttl, loc="left", fontsize=8)
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_ylim(0, 1.2)
    save(fig, "parameter_curves")


def fig_tv_convergence():
    d = pd.read_csv(RES / "tv_convergence.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    t = 0
    for ph in ["digits", "shepp_logan"]:
        for m in (10, 180):
            g = d[(d.phantom == ph) & (d.angles == m)].sort_values("iter")
            fref = g.objective.iloc[-1]                     # value after the last (3000th) iteration
            lab = f"{LABEL[ph]}, {m} angles"
            h = g[g.iter <= 2000]
            axes[0].semilogy(h.iter, (h.objective - fref) / fref, c=CB[t], label=lab)
            axes[1].plot(g.iter, g.rel_err, c=CB[t], label=lab)
            t += 1
    for ax in axes:
        ax.axvline(600, c="#999999", lw=0.8)
        ax.set_xlabel("iteration")
    axes[0].set_ylabel(r"$(\Phi(x^k)-\Phi(x^{3000}))/\Phi(x^{3000})$")
    axes[1].set_ylabel("relative error")
    axes[0].set_title("a) objective", loc="left", fontsize=8)
    axes[1].set_title("b) reconstruction error", loc="left", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7)
    save(fig, "tv_convergence")


def main():
    for f in (fig_spectrum, fig_nullspace, fig_error_vs_angles, fig_reconstructions, fig_parameters,
              fig_tv_convergence):
        try:
            f()
        except FileNotFoundError as e:
            print(f"skipped {f.__name__}: {e}")


if __name__ == "__main__":
    main()
