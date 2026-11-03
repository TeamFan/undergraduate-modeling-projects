"""Medical-supply warehouse location and helicopter routing (SWPU 2023 campus contest, problem A).

All functions are deterministic given their seeds.  Distances are great-circle
distances in km; times are in minutes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp, minimize

DATA = Path(__file__).resolve().parent.parent / "data"

EARTH_RADIUS = 6371.0  # km
RANGE = 2000.0         # Mi-26 maximum range, km
PAYLOAD = 12000.0      # Mi-26 maximum payload, kg
SPEED = 255.0          # Mi-26 cruise speed, km/h
FLEET = 10             # helicopters available at the base

# Bounding box of Sichuan used as the GA search domain (degrees).
LON_BOUNDS = (97.35, 108.55)
LAT_BOUNDS = (26.05, 34.32)


# --------------------------------------------------------------------- data
def load_data():
    loc = pd.read_excel(DATA / "attachment1_hospital_locations.xls")
    loc.columns = ["city", "lon", "lat"]
    dem = pd.read_excel(DATA / "attachment2_monthly_demand.xls").iloc[:, :2]
    dem.columns = ["city", "demand"]
    cities = loc.merge(dem, on="city", validate="one_to_one")
    loss = pd.read_excel(DATA / "attachment3_loss_rate.xls")
    loss.columns = ["minutes", "rate"]
    return cities, loss


# --------------------------------------------------------------------- geometry
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def total_distance(point, cities, weights=None):
    d = haversine(point[0], point[1], cities["lon"].to_numpy(), cities["lat"].to_numpy())
    return float(d.sum() if weights is None else d @ weights)


def distance_matrix(base, cities):
    lon = np.r_[base[0], cities["lon"].to_numpy()]
    lat = np.r_[base[1], cities["lat"].to_numpy()]
    return haversine(lon[:, None], lat[:, None], lon[None, :], lat[None, :])


# --------------------------------------------------------------------- problem 1
def genetic_algorithm(f, seed, pop=100, gens=200, pc=0.5, pm=0.1, b=2.0):
    """Real-coded GA: roulette selection on 1/f,
    arithmetic crossover and Michalewicz non-uniform mutation, with elitism."""
    rng = np.random.default_rng(seed)
    lo = np.array([LON_BOUNDS[0], LAT_BOUNDS[0]])
    hi = np.array([LON_BOUNDS[1], LAT_BOUNDS[1]])
    X = lo + rng.random((pop, 2)) * (hi - lo)
    fx = np.array([f(x) for x in X])
    history = [fx.min()]
    for g in range(gens):
        elite = X[fx.argmin()].copy()
        fit = 1.0 / fx
        idx = rng.choice(pop, size=pop, p=fit / fit.sum())
        X = X[idx].copy()
        for i in range(0, pop - 1, 2):
            if rng.random() < pc:
                a = rng.random()
                xa, xb = X[i].copy(), X[i + 1].copy()
                X[i], X[i + 1] = a * xb + (1 - a) * xa, a * xa + (1 - a) * xb
        r = rng.random(X.shape)
        step = 1 - r ** ((1 - g / gens) ** b)
        up = rng.random(X.shape) < 0.5
        mutate = rng.random(X.shape) < pm
        X = np.where(mutate & up, X + (hi - X) * step, X)
        X = np.where(mutate & ~up, X - (X - lo) * step, X)
        X[0] = elite
        fx = np.array([f(x) for x in X])
        history.append(fx.min())
    return X[fx.argmin()], float(fx.min()), np.array(history)


def weber_point(cities, weights=None, starts=None):
    """Local refinement of the spherical Weber point from several starts."""
    f = lambda p: total_distance(p, cities, weights)
    if starts is None:
        starts = cities[["lon", "lat"]].to_numpy()
    best = None
    for s in starts:
        res = minimize(f, s, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)


# --------------------------------------------------------------------- loss rate
def fit_loss_curve(loss):
    """Fit rate = c * (t/300)^p by least squares on the tabulated points."""
    t, y = loss["minutes"].to_numpy(float), loss["rate"].to_numpy(float)
    sse = lambda q: np.sum((q[0] * (t / 300.0) ** q[1] - y) ** 2)
    res = minimize(sse, [0.1, 3.0], method="Nelder-Mead", options={"xatol": 1e-12, "fatol": 1e-16})
    c, p = res.x
    return float(c), float(p), float(np.max(np.abs(c * (t / 300.0) ** p - y)))


def loss_rate(minutes, c, p):
    return c * (np.asarray(minutes) / 300.0) ** p


# --------------------------------------------------------------------- routing MILP
def solve_routing(D, q, k=None, max_k=FLEET, distance_cap=None, loss=None, time_limit=300):
    """Capacitated, range-limited vehicle routing MILP (two-index flow, MTZ).

    D        : (n+1)x(n+1) distance matrix, node 0 is the base.
    q        : demands of the n hospitals (kg).
    k        : fixed number of helicopters, or None for 1..max_k.
    distance_cap : if given, total flight distance <= distance_cap.
    loss     : (c, p) to minimise the total in-flight loss of supplies instead of
               distance.  The loss of hospital j is q_j * rate(arrival time at j);
               the convex rate is represented by tangent cuts (outer approximation)
               and the exact loss of the returned routes is recomputed afterwards.
    Returns a dict with routes, total distance, solver status and MIP gap.
    """
    n = len(q)
    N = n + 1
    arcs = [(i, j) for i in range(N) for j in range(N) if i != j]
    na = len(arcs)
    aidx = {a: t for t, a in enumerate(arcs)}
    # variable layout: x (na) | u (n, arrival distance) | g (n, cumulative load) | l (n, loss)
    ux = na
    gx = na + n
    lx = na + 2 * n
    nv = na + 3 * n

    rows, lo, hi = [], [], []

    def row():
        return np.zeros(nv)

    for j in range(1, N):  # in-degree 1
        r = row()
        for i in range(N):
            if i != j:
                r[aidx[(i, j)]] = 1
        rows.append(r); lo.append(1); hi.append(1)
    for i in range(1, N):  # out-degree 1
        r = row()
        for j in range(N):
            if j != i:
                r[aidx[(i, j)]] = 1
        rows.append(r); lo.append(1); hi.append(1)
    kmin, kmax = (k, k) if k is not None else (1, max_k)
    r = row()
    for j in range(1, N):
        r[aidx[(0, j)]] = 1
    rows.append(r); lo.append(kmin); hi.append(kmax)
    r = row()
    for i in range(1, N):
        r[aidx[(i, 0)]] = 1
    rows.append(r); lo.append(kmin); hi.append(kmax)

    Mu = RANGE + D.max()
    for i in range(1, N):
        for j in range(1, N):
            if i == j:
                continue
            # u_j >= u_i + d_ij - M(1 - x_ij)
            r = row(); r[ux + j - 1] = 1; r[ux + i - 1] = -1; r[aidx[(i, j)]] = -Mu
            rows.append(r); lo.append(D[i, j] - Mu); hi.append(np.inf)
            # g_j >= g_i + q_j - Q(1 - x_ij)
            r = row(); r[gx + j - 1] = 1; r[gx + i - 1] = -1; r[aidx[(i, j)]] = -PAYLOAD
            rows.append(r); lo.append(q[j - 1] - PAYLOAD); hi.append(np.inf)
    # range: u_i + d_i0 <= RANGE when (i, 0) is used
    for i in range(1, N):
        r = row(); r[ux + i - 1] = 1; r[aidx[(i, 0)]] = Mu
        rows.append(r); lo.append(-np.inf); hi.append(RANGE - D[i, 0] + Mu)
    if distance_cap is not None:
        r = row()
        for t, (i, j) in enumerate(arcs):
            r[t] = D[i, j]
        rows.append(r); lo.append(-np.inf); hi.append(distance_cap)

    c = np.zeros(nv)
    if loss is None:
        for t, (i, j) in enumerate(arcs):
            c[t] = D[i, j]
    else:
        cl, p = loss
        km_per_min = SPEED / 60.0
        for j in range(1, N):
            c[lx + j - 1] = 1.0
            for uk in np.linspace(0, RANGE, 81):
                tk = uk / km_per_min
                f = q[j - 1] * loss_rate(tk, cl, p)
                df = q[j - 1] * cl * p * (tk / 300.0) ** (p - 1) / 300.0 / km_per_min
                # l_j >= f + df * (u_j - uk)
                r = row(); r[lx + j - 1] = 1; r[ux + j - 1] = -df
                rows.append(r); lo.append(f - df * uk); hi.append(np.inf)
        # tiny distance term breaks ties between plans with equal loss
        for t, (i, j) in enumerate(arcs):
            c[t] = 1e-6 * D[i, j]

    lb = np.zeros(nv); ub = np.ones(nv)
    lb[ux:ux + n] = D[0, 1:]; ub[ux:ux + n] = RANGE - D[1:, 0]
    lb[gx:gx + n] = q; ub[gx:gx + n] = PAYLOAD
    lb[lx:] = 0; ub[lx:] = np.inf
    integrality = np.zeros(nv); integrality[:na] = 1

    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality,
               bounds=Bounds(lb, ub), options={"time_limit": time_limit, "mip_rel_gap": 1e-6, "disp": False})
    if res.x is None:
        return {"status": res.status, "message": res.message, "routes": None}
    x = np.round(res.x[:na]).astype(int)
    succ = {}
    for t, (i, j) in enumerate(arcs):
        if x[t]:
            succ.setdefault(i, []).append(j)
    routes = []
    for start in succ[0]:
        route, cur = [], start
        while cur != 0:
            route.append(cur)
            cur = succ[cur][0]
        routes.append(route)
    return {
        "status": res.status,
        "message": res.message,
        "routes": routes,
        "distance": route_distance(D, routes),
        "mip_gap": getattr(res, "mip_gap", None),
        "objective": float(res.fun),
        "bound": getattr(res, "mip_dual_bound", None),
    }


def served_at_base(D, routes, tol=1e-9):
    """Remove stops that coincide with the base (zero distance): their supplies
    are handed over at take-off, so they need no flight and suffer no loss.
    Routes that become empty are dropped."""
    out = [[j for j in r if D[0, j] > tol] for r in routes]
    return [r for r in out if r]


def route_distance(D, routes):
    total = 0.0
    for r in routes:
        path = [0] + list(r) + [0]
        total += sum(D[a, b] for a, b in zip(path[:-1], path[1:]))
    return float(total)


def route_lengths(D, routes):
    return [route_distance(D, [r]) for r in routes]


def route_loss(D, routes, q, c, p):
    """Exact in-flight loss (kg) of a set of routes; goods for hospital j stay
    on board from take-off until the helicopter reaches j."""
    total = 0.0
    km_per_min = SPEED / 60.0
    for r in routes:
        u = 0.0
        prev = 0
        for j in r:
            u += D[prev, j]
            total += q[j - 1] * loss_rate(u / km_per_min, c, p)
            prev = j
    return float(total)


def direct_loss(D, q, c, p):
    """Lower bound on loss: every hospital served by its own direct flight."""
    t = D[0, 1:] / (SPEED / 60.0)
    return float(np.sum(q * loss_rate(t, c, p)))
