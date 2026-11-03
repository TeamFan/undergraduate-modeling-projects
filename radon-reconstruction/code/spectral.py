"""Spectral analysis of the projection matrices on the 100 x 100 grid: rank, null
space, conditioning, the null-space error bound of Proposition 3.1 and the
discrete Picard plot.

    python code/spectral.py

Writes results/system_sizes.csv, results/singular_values.csv, results/picard.csv,
results/nullspace_bound.csv and results/nullspace_digits_10.npz."""

import time

import numpy as np
import pandas as pd

from common import ANGLES, N, PHANTOMS, RES, add_noise, coarse, make_data, phantoms_fine
from tomo import angles, system_matrix

RANK_TOL = 1e-7          # singular values below RANK_TOL * sigma_1 are treated as zero (eigenvalue accuracy ~1e-8)


def spectrum(A):
    """Singular values and an orthonormal basis of range(A^T), from the smaller Gram matrix.
    Returns (sigma descending, V with columns = right singular vectors for sigma > 0 or None, U or None)."""
    M, n = A.shape
    if M < n:
        lam, U = np.linalg.eigh((A @ A.T).toarray())
        lam, U = lam[::-1], U[:, ::-1]
        s = np.sqrt(np.clip(lam, 0, None))
        return s, None, U
    lam, V = np.linalg.eigh((A.T @ A).toarray())
    lam, V = lam[::-1], V[:, ::-1]
    return np.sqrt(np.clip(lam, 0, None)), V, None


def main():
    t0 = time.time()
    ph = phantoms_fine()
    truth = {k: coarse(v) for k, v in ph.items()}
    sizes, svals, bound, picard = [], [], [], []
    for m in ANGLES:
        A, nd = system_matrix(N, angles(m))
        s, V, U = spectrum(A)
        r = int(np.sum(s > RANK_TOL * s[0]))
        gap = float(s[r] / s[r - 1]) if r < s.size else 0.0
        sizes.append({"angles": m, "equations": A.shape[0], "unknowns": A.shape[1],
                      "type": "overdetermined" if A.shape[0] > A.shape[1] else "underdetermined",
                      "rank": r, "nullity": A.shape[1] - r, "sigma_1": float(s[0]), "sigma_r": float(s[r - 1]),
                      "cond": float(s[0] / s[r - 1]), "gap_below_rank": gap})
        svals.append(pd.DataFrame({"angles": m, "i": np.arange(1, s.size + 1), "sigma": s}))
        for name in PHANTOMS:
            x = truth[name].ravel()
            b = make_data(ph[name], m, nd)
            if U is not None:                      # underdetermined: range(A^T) = A^T U_r
                Ur, sr = U[:, :r], s[:r]
                Vr = (A.T @ Ur) / sr               # right singular vectors, n x r
                xR = Vr @ (Vr.T @ x)
                ub = Ur.T @ b                      # u_i^T b
            else:
                Vr, sr = V[:, :r], s[:r]
                xR = Vr @ (Vr.T @ x)
                ub = (A @ Vr).T @ b / sr
            xN = x - xR
            ls_floor = np.sqrt(max(np.linalg.norm(b) ** 2 - np.sum(ub ** 2), 0)) / np.linalg.norm(b)
            bound.append({"phantom": name, "angles": m, "nullity": A.shape[1] - r,
                          "null_fraction": float(np.linalg.norm(xN) / np.linalg.norm(x)),
                          "ls_residual_floor": float(ls_floor)})
            if name == "digits" and m == 10:
                np.savez_compressed(RES / "nullspace_digits_10.npz", truth=x.reshape(N, N),
                                    range_part=xR.reshape(N, N), null_part=xN.reshape(N, N))
            if m == 180:
                ux = (A @ Vr).T @ (A @ x) / sr     # coefficients of the operator-consistent data A x_true
                df = pd.DataFrame({"phantom": name, "i": np.arange(1, r + 1), "sigma": sr,
                                   "coef_consistent": np.abs(ux), "coef_fine": np.abs(ub)})
                for eta in (0.01, 0.05):
                    bn, _ = add_noise(b, eta, m, 0)
                    df[f"coef_noise{int(100 * eta)}"] = np.abs((A @ Vr).T @ bn / sr)
                picard.append(df)
        print(f"[{time.strftime('%H:%M:%S')}] m={m}: rank {r}, cond {s[0] / s[r - 1]:.3g}", flush=True)
    pd.DataFrame(sizes).to_csv(RES / "system_sizes.csv", index=False)
    pd.concat(svals).to_csv(RES / "singular_values.csv", index=False, float_format="%.6g")
    pd.DataFrame(bound).to_csv(RES / "nullspace_bound.csv", index=False)
    pd.concat(picard).to_csv(RES / "picard.csv", index=False, float_format="%.6g")
    print(f"spectral analysis finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
