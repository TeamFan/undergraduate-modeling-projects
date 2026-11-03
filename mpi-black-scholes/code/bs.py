"""Closed-form prices, Monte Carlo path kernels and a Crank-Nicolson solver
for the Black-Scholes model (European and discretely monitored Asian calls)."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
from scipy.stats import norm


# ------------------------------------------------------------------ closed forms
def bs_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_greeks(S0, K, T, r, sigma):
    """Delta and gamma of the European call (S0 may be an array)."""
    S0 = np.asarray(S0, float)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1), norm.pdf(d1) / (S0 * sigma * np.sqrt(T))


def geometric_asian_call(S0, K, T, r, sigma, M):
    """Call on the geometric mean of S(t_1), ..., S(t_M), t_i = i T / M.
    log G is Gaussian with the mean and variance below (Kemna and Vorst, 1990)."""
    dt = T / M
    mu = np.log(S0) + (r - 0.5 * sigma ** 2) * dt * (M + 1) / 2
    s2 = sigma ** 2 * dt * (M + 1) * (2 * M + 1) / (6 * M)
    s = np.sqrt(s2)
    d1 = (mu - np.log(K) + s2) / s
    d2 = d1 - s
    return np.exp(-r * T) * (np.exp(mu + 0.5 * s2) * norm.cdf(d1) - K * norm.cdf(d2))


# ------------------------------------------------------------------ Monte Carlo kernel
def simulate(rng, n, S0, K, T, r, sigma, M, antithetic=False):
    """Simulate n GBM paths with M log-Euler steps (exact in distribution at the
    grid points) and return discounted samples, one entry per path or per
    antithetic pair:

        euro   : e^{-rT} (S_T - K)^+
        st     : e^{-rT} S_T                     (control, mean S0)
        asian  : e^{-rT} (A - K)^+,  A = mean_i S(t_i)
        geo    : e^{-rT} (G - K)^+,  G = geometric mean   (control, closed form)

    Memory is O(n); the work is O(n M)."""
    dt = T / M
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * np.sqrt(dt)
    signs = (1.0, -1.0) if antithetic else (1.0,)
    m = n // len(signs)
    logS = [np.full(m, np.log(S0)) for _ in signs]
    sumS = [np.zeros(m) for _ in signs]
    sumlog = [np.zeros(m) for _ in signs]
    for _ in range(M):
        Z = rng.standard_normal(m)
        for k, s in enumerate(signs):
            logS[k] += drift + vol * s * Z
            sumS[k] += np.exp(logS[k])
            sumlog[k] += logS[k]
    disc = np.exp(-r * T)
    out = {}
    for name, f in [("euro", lambda k: np.maximum(np.exp(logS[k]) - K, 0.0)),
                    ("st", lambda k: np.exp(logS[k])),
                    ("asian", lambda k: np.maximum(sumS[k] / M - K, 0.0)),
                    ("geo", lambda k: np.maximum(np.exp(sumlog[k] / M) - K, 0.0))]:
        out[name] = disc * np.mean([f(k) for k in range(len(signs))], axis=0)
    return out


def moments(samples):
    """Additive sufficient statistics (count, sums, cross sums) so that
    estimates from any partition of the paths can be merged exactly."""
    x = {k: np.asarray(v, float) for k, v in samples.items()}
    st = {"n": x["euro"].size}
    for k, v in x.items():
        st[f"s_{k}"] = float(v.sum())
        st[f"ss_{k}"] = float(np.sum(v * v))
    st["sp_euro_st"] = float(np.sum(x["euro"] * x["st"]))
    st["sp_asian_geo"] = float(np.sum(x["asian"] * x["geo"]))
    return st


def merge(stats):
    out = {}
    for s in stats:
        for k, v in s.items():
            out[k] = out.get(k, 0) + v
    return out


def estimates(st, S0, K, T, r, sigma, M):
    """Plain and control-variate estimators with standard errors.  The CV
    coefficient beta = Cov(X, Y)/Var(Y) is estimated from the same samples
    (its O(1/n) bias is negligible at the sample sizes used here)."""
    n = st["n"]

    def mean_var(a):
        mu = st[f"s_{a}"] / n
        return mu, (st[f"ss_{a}"] - n * mu * mu) / (n - 1)

    def cov(a, b, key):
        return (st[key] - n * (st[f"s_{a}"] / n) * (st[f"s_{b}"] / n)) / (n - 1)

    res = {"n": n}
    for x, y, key, ey in [("euro", "st", "sp_euro_st", S0),
                          ("asian", "geo", "sp_asian_geo", geometric_asian_call(S0, K, T, r, sigma, M))]:
        mx, vx = mean_var(x)
        my, vy = mean_var(y)
        cxy = cov(x, y, key)
        beta = cxy / vy
        res[x] = {"plain": mx, "plain_se": np.sqrt(vx / n),
                  "cv": mx - beta * (my - ey), "cv_se": np.sqrt(max(vx - cxy ** 2 / vy, 0.0) / n),
                  "beta": beta, "corr": cxy / np.sqrt(vx * vy)}
    return res


# ------------------------------------------------------------------ Crank-Nicolson
def cn_grid(K, T, r, sigma, Smax, M=250, N=2000, rannacher=2):
    """Crank-Nicolson solution of the Black-Scholes equation for a European call
    in tau = T - t on the uniform grid S_j = j Smax / N, j = 0..N.

    rannacher: number of initial fully implicit half-steps (Rannacher, 1984)
    that damp the oscillations caused by the kink of the payoff; 0 gives plain CN.
    Returns the grid S and the values V(T, S_j) at time to maturity T.
    """
    S = np.linspace(0.0, Smax, N + 1)
    i = np.arange(1, N)
    V = np.maximum(S - K, 0.0)
    upper = lambda tau: Smax - K * np.exp(-r * tau)
    # semi-discrete operator (L V)_i = a V_{i-1} + b V_i + c V_{i+1}
    a = 0.5 * (sigma ** 2 * i ** 2 - r * i)
    b = -(sigma ** 2 * i ** 2 + r)
    c = 0.5 * (sigma ** 2 * i ** 2 + r * i)

    def step(V, tau, h, theta):
        """theta-scheme step of size h: (I - theta h L) V^{new} = (I + (1-theta) h L) V^{old}."""
        A = np.zeros((3, N - 1))
        A[0, 1:] = -theta * h * c[:-1]
        A[1, :] = 1 - theta * h * b
        A[2, :-1] = -theta * h * a[1:]
        e = (1 - theta) * h
        rhs = V[1:-1] + e * (a * V[:-2] + b * V[1:-1] + c * V[2:])
        rhs[-1] += theta * h * c[-1] * upper(tau + h) + 0.0
        Vn = V.copy()
        Vn[1:-1] = solve_banded((1, 1), A, rhs)
        Vn[0], Vn[-1] = 0.0, upper(tau + h)
        return Vn

    dt = T / M
    tau = 0.0
    for _ in range(rannacher):           # implicit Euler half-steps
        V = step(V, tau, dt / 2, 1.0)
        tau += dt / 2
    nsteps = M - rannacher // 2
    for _ in range(nsteps):
        V = step(V, tau, dt, 0.5)
        tau += dt
    return S, V


def cn_call(S0, K, T, r, sigma, M=250, N=2000, smax=4.0, rannacher=2):
    """European call by Crank-Nicolson on S in [0, smax*S0] (see cn_grid).
    The price at S0 is read by cubic interpolation, so N need not place S0 on a node."""
    S, V = cn_grid(K, T, r, sigma, smax * S0, M=M, N=N, rannacher=rannacher)
    return float(CubicSpline(S, V)(S0))
