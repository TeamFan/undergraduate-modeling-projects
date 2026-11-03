"""Model checks and sensitivity analyses that complement run_all.py.

    python code/further_analysis.py fast    # Q1 landscape, Q2 routes, heuristic check, Q3 trade-off (~2 min)
    python code/further_analysis.py slow    # extra epsilon points and range/payload grid (~30 min)
    python code/further_analysis.py         # both, then the summary of the trade-off

The script reads results/results.json (written by run_all.py q123) for the
distance-optimal and loss-optimal plans, and writes CSV/JSON files with the
prefix of the problem they belong to into results/, figures into results/figures/.
"""

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import model
from model import (FLEET, PAYLOAD, RANGE, SPEED, distance_matrix, fit_loss_curve, genetic_algorithm,
                   haversine, load_data, loss_rate, route_distance, route_loss, solve_routing,
                   total_distance, weber_point)
from run_all import CB, EN, FIG, RES, draw_map, log, plt, save, tangent_gradient, vertex_condition

EPS_EXTRA = [0.15, 0.20, 0.30, 0.35, 0.40, 0.75]


def setup():
    cities, loss_table = load_data()
    cities["name_en"] = cities["city"].map(EN)
    q = cities["demand"].to_numpy(float)
    c, p, _ = fit_loss_curve(loss_table)
    weber, _ = weber_point(cities)
    k = int(np.argmin(haversine(weber[0], weber[1], cities["lon"], cities["lat"])))
    base = cities.loc[k, ["lon", "lat"]].to_numpy(float)
    D = distance_matrix(base, cities)
    res = json.loads((RES / "results.json").read_text())
    idx = {n: i + 1 for i, n in enumerate(cities["name_en"])}
    return cities, loss_table, q, c, p, k, base, D, res, idx


def to_nodes(routes, idx):
    return [[idx[n] for n in r] for r in routes]


def destination(lon, lat, bearing, dist):
    """Point reached from (lon, lat) after `dist` km along a great circle with `bearing` (rad)."""
    lam, phi = np.radians(lon), np.radians(lat)
    delta = dist / model.EARTH_RADIUS
    phi2 = np.arcsin(np.sin(phi) * np.cos(delta) + np.cos(phi) * np.sin(delta) * np.cos(bearing))
    lam2 = lam + np.arctan2(np.sin(bearing) * np.sin(delta) * np.cos(phi), np.cos(delta) - np.sin(phi) * np.sin(phi2))
    return np.degrees(lam2), np.degrees(phi2)


# ------------------------------------------------------------------ problem 1
def q1_landscape(cities, k, base):
    f = lambda z: total_distance(z, cities)
    fstar = f(base)
    out = {}
    # Kuhn's vertex test at every hospital site
    rows = []
    for j in range(len(cities)):
        norm, w = vertex_condition(j, cities)
        rows.append({"city": cities.loc[j, "name_en"], "lon": cities.loc[j, "lon"], "lat": cities.loc[j, "lat"],
                     "resultant_norm": norm, "optimal": norm <= w, "total_km": f(cities.loc[j, ["lon", "lat"]].to_numpy(float))})
    vt = pd.DataFrame(rows).sort_values("resultant_norm")
    vt.to_csv(RES / "q1_vertex_test.csv", index=False)
    # resultant of the unit vectors of the other hospitals at Ziyang (east, north)
    others = cities.drop(index=k)
    S = tangent_gradient(base, others)
    out["resultant_east_north"] = S.tolist()
    out["resultant_norm"] = float(np.linalg.norm(S))
    out["slope_min"] = float(1 - np.linalg.norm(S))
    out["slope_max"] = float(1 + np.linalg.norm(S))
    out["cheapest_direction_bearing_deg"] = float(np.degrees(np.arctan2(-S[0], -S[1])) % 360)
    # excess total distance on circles around Ziyang
    th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    flat = []
    for r in [1, 2, 5, 10, 20, 50, 100]:
        ex = np.array([f(destination(base[0], base[1], t, r)) - fstar for t in th])
        flat.append({"radius_km": r, "min_excess_km": ex.min(), "max_excess_km": ex.max(),
                     "first_order_min_km": out["slope_min"] * r, "first_order_max_km": out["slope_max"] * r,
                     "bearing_of_min_deg": float(np.degrees(th[ex.argmin()]))})
    fl = pd.DataFrame(flat)
    fl.to_csv(RES / "q1_flatness.csv", index=False)
    # naive planar formulations that are common shortcuts
    pts = cities[["lon", "lat"]].to_numpy(float)
    naive = minimize(lambda z: np.sqrt(((pts - z) ** 2).sum(1)).sum(), pts.mean(0), method="Nelder-Mead",
                     options={"xatol": 1e-10, "fatol": 1e-10}).x
    s = np.array([np.cos(np.radians(pts[:, 1].mean())), 1.0])
    equi = minimize(lambda z: np.sqrt((((pts - z) * s) ** 2).sum(1)).sum(), pts.mean(0), method="Nelder-Mead",
                    options={"xatol": 1e-10, "fatol": 1e-10}).x
    out["planar_degrees"] = {"lon": naive[0], "lat": naive[1], "total_km": f(naive), "excess_km": f(naive) - fstar,
                             "offset_km": float(haversine(*naive, *base))}
    out["equirectangular"] = {"lon": equi[0], "lat": equi[1], "total_km": f(equi), "excess_km": f(equi) - fstar,
                              "offset_km": float(haversine(*equi, *base))}
    # GA control parameters
    ga = []
    for pop in [50, 100, 200]:
        for pm in [0.05, 0.1, 0.2]:
            for seed in range(10):
                x, fx, _ = genetic_algorithm(f, seed, pop=pop, pm=pm)
                ga.append({"pop": pop, "pm": pm, "seed": seed, "total_km": fx, "offset_km": float(haversine(*x, *base))})
    ga = pd.DataFrame(ga)
    ga.to_csv(RES / "q1_ga_parameters.csv", index=False)
    gs = ga.groupby(["pop", "pm"]).agg(mean_offset=("offset_km", "mean"), max_offset=("offset_km", "max"),
                                       mean_excess=("total_km", lambda v: v.mean() - fstar)).reset_index()
    out["ga_parameters"] = gs.to_dict(orient="records")
    # contour figure
    lon = np.linspace(base[0] - 1.6, base[0] + 1.6, 161)
    lat = np.linspace(base[1] - 1.3, base[1] + 1.3, 131)
    LON, LAT = np.meshgrid(lon, lat)
    lo, la = cities["lon"].to_numpy()[None, None, :], cities["lat"].to_numpy()[None, None, :]
    F = haversine(LON[..., None], LAT[..., None], lo, la).sum(-1) - fstar
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    cs = ax.contour(LON, LAT, F, levels=[5, 10, 25, 50, 100, 200, 400], colors="#555555", linewidths=0.7)
    ax.clabel(cs, fmt="%d km", fontsize=6)
    inside = (cities.lon.between(lon[0], lon[-1])) & (cities.lat.between(lat[0], lat[-1]))
    ax.scatter(cities.lon[inside], cities.lat[inside], s=12, c="#555555", zorder=3)
    for _, r in cities[inside].iterrows():
        ax.annotate(r["name_en"], (r.lon, r.lat), xytext=(3, 2), textcoords="offset points", fontsize=7)
    ax.plot(*base, marker="*", ms=13, c=CB[1], ls="none", zorder=4, label="Weber point (Ziyang)")
    ax.plot(*naive, marker="^", ms=10, mfc="none", mec=CB[0], mew=1.3, ls="none", zorder=5,
            label="planar Weber point in degrees")
    ax.annotate("", xy=destination(base[0], base[1], np.radians(out["cheapest_direction_bearing_deg"]), 40),
                xytext=base, arrowprops=dict(arrowstyle="->", color=CB[2], lw=1.2))
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.set_aspect(1 / np.cos(np.radians(30)))
    ax.legend(loc="upper left", fontsize=7)
    save(fig, "q1_contours")
    return out, vt, fl


# ------------------------------------------------------------------ problem 2
def route_table(D, routes, q, c, p, names):
    rows = []
    v = SPEED / 60.0
    for h, r in enumerate(routes, 1):
        u, prev, load = 0.0, 0, sum(q[j - 1] for j in r)
        for j in r:
            u += D[prev, j]
            rows.append({"helicopter": h, "hospital": names[j - 1], "leg_km": D[prev, j], "cumulative_km": u,
                         "arrival_min": u / v, "demand_kg": q[j - 1], "on_board_before_kg": load,
                         "loss_kg": q[j - 1] * loss_rate(u / v, c, p)})
            load -= q[j - 1]
            prev = j
        rows.append({"helicopter": h, "hospital": "(return)", "leg_km": D[prev, 0], "cumulative_km": u + D[prev, 0],
                     "arrival_min": (u + D[prev, 0]) / v, "demand_kg": 0.0, "on_board_before_kg": 0.0, "loss_kg": 0.0})
    return pd.DataFrame(rows)


def orientation_losses(D, routes, q, c, p):
    out = []
    for r in routes:
        fwd = route_loss(D, [r], q, c, p)
        bwd = route_loss(D, [r[::-1]], q, c, p)
        out.append({"forward_kg": fwd, "reverse_kg": bwd, "length_km": route_distance(D, [r])})
    return out


def savings_heuristic(D, q, nodes, rng_limit=RANGE, cap=PAYLOAD):
    """Clarke-Wright parallel savings followed by 2-opt, relocate and swap moves."""
    routes = {j: [j] for j in nodes}
    where = {j: j for j in nodes}
    s = sorted(((D[0, i] + D[0, j] - D[i, j], i, j) for i, j in combinations(nodes, 2)), reverse=True)
    for sav, i, j in s:
        if sav <= 0:
            break
        ri, rj = where[i], where[j]
        if ri == rj:
            continue
        a, b = routes[ri], routes[rj]
        merged = None
        for x in (a, a[::-1]):
            for y in (b, b[::-1]):
                if x[-1] == i and y[0] == j or x[-1] == j and y[0] == i:
                    merged = x + y
        if merged is None:
            continue
        if sum(q[t - 1] for t in merged) > cap or route_distance(D, [merged]) > rng_limit:
            continue
        routes[ri] = merged
        del routes[rj]
        for t in merged:
            where[t] = ri
    R = [list(r) for r in routes.values()]

    def ok(r):
        return sum(q[t - 1] for t in r) <= cap and route_distance(D, [r]) <= rng_limit + 1e-9

    def two_opt(r):
        best, improved = r, True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 1, len(best)):
                    cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    if route_distance(D, [cand]) < route_distance(D, [best]) - 1e-9:
                        best, improved = cand, True
        return best

    R = [two_opt(r) for r in R]
    improved = True
    while improved:
        improved = False
        total = route_distance(D, R)
        for a in range(len(R)):
            for b in range(len(R)):
                if a == b:
                    continue
                for i, x in enumerate(R[a]):
                    for pos in range(len(R[b]) + 1):  # relocate x from route a to route b
                        na = R[a][:i] + R[a][i + 1:]
                        nb = R[b][:pos] + [x] + R[b][pos:]
                        if ok(nb):
                            new = [r for t, r in enumerate(R) if t not in (a, b)] + ([na] if na else []) + [nb]
                            if route_distance(D, new) < total - 1e-9:
                                R, improved = [two_opt(r) for r in new], True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return R


def q2_checks(D, q, c, p, res, idx, names, k):
    out = {}
    plan = to_nodes(res["q3"]["front"][0]["routes"], idx)  # distance-optimal plan, Ziyang served at the depot
    rt = route_table(D, plan, q, c, p, names)
    rt.to_csv(RES / "q2_route_detail.csv", index=False)
    out["plan_distance_km"] = route_distance(D, plan)
    out["plan_loss_kg"] = route_loss(D, plan, q, c, p)
    out["orientation"] = orientation_losses(D, plan, q, c, p)
    out["max_round_trip_km"] = float(2 * D[0, 1:].max())
    out["farthest_hospital"] = names[int(D[0, 1:].argmax())]
    nodes = [j for j in range(1, len(q) + 1) if j != k + 1]
    t0 = time.time()
    H = savings_heuristic(D, q, nodes)
    out["savings_heuristic"] = {"distance_km": route_distance(D, H), "routes": [[names[j - 1] for j in r] for r in H],
                                "gap_to_milp": route_distance(D, H) / out["plan_distance_km"] - 1,
                                "runtime_s": time.time() - t0}
    # speed scaling: for fixed routes the loss scales with (v0/v)^p
    out["speed_scaling"] = [{"speed_kmh": v, "distance_optimal_kg": out["plan_loss_kg"] * (SPEED / v) ** p,
                             "loss_optimal_kg": res["q3"]["front"][-1]["loss_kg"] * (SPEED / v) ** p}
                            for v in [200.0, 255.0, 300.0, 400.0]]
    return out, rt


def q2_range_payload(D, q, time_limit=180):
    rows = []
    cases = [(r, 12000.0) for r in [1000.0, 1200.0, 1500.0, 1750.0, 2500.0]] + \
            [(2000.0, w) for w in [3000.0, 4000.0, 6000.0, 8000.0]]
    for rng, cap in cases:
        model.RANGE, model.PAYLOAD = rng, cap
        r = solve_routing(D, q, time_limit=time_limit)
        fly = [x for x in (r["routes"] or []) if route_distance(D, [x]) > 1e-9]
        rows.append({"range_km": rng, "payload_kg": cap, "status": r["status"], "mip_gap": r.get("mip_gap"),
                     "distance_km": r.get("distance"), "helicopters": len(fly),
                     "longest_route_km": max(route_distance(D, [x]) for x in fly) if fly else None,
                     "max_load_kg": max(sum(q[j - 1] for j in x) for x in fly) if fly else None})
        log(f"  range {rng:.0f} payload {cap:.0f}: {rows[-1]}")
    model.RANGE, model.PAYLOAD = RANGE, PAYLOAD
    pd.DataFrame(rows).to_csv(RES / "q2_range_payload.csv", index=False)


# ------------------------------------------------------------------ problem 3
def q3_extra_eps(D, q, c, p, names, dmin):
    rows = []
    for eps in EPS_EXTRA:
        r = solve_routing(D, q, loss=(c, p), distance_cap=dmin * (1 + eps) + 1e-6)
        fly = [x for x in r["routes"] if route_distance(D, [x]) > 1e-9]
        rows.append({"eps": eps, "distance_km": r["distance"], "loss_kg": route_loss(D, r["routes"], q, c, p),
                     "helicopters": len(fly), "status": r["status"], "mip_gap": r.get("mip_gap"),
                     "max_arrival_min": max(route_distance(D, [x]) - D[x[-1], 0] for x in fly) / SPEED * 60,
                     "routes": [[names[j - 1] for j in x] for x in fly]})
        log(f"  eps={eps}: D={rows[-1]['distance_km']:.1f} loss={rows[-1]['loss_kg']:.2f} k={len(fly)}")
    (RES / "q3_front_extra.json").write_text(json.dumps(rows, indent=2, default=float))


def q3_tradeoff(D, q, c, p, res, idx, names):
    front = [dict(r) for r in res["q3"]["front"]]
    extra = RES / "q3_front_extra.json"
    if extra.exists():
        front += json.loads(extra.read_text())
    fr = pd.DataFrame(front)
    fr["eps_label"] = fr.eps.map(lambda e: "none" if pd.isna(e) else f"{e:.2f}")
    fr = fr.sort_values("distance_km").drop_duplicates(subset=["distance_km", "loss_kg"]).reset_index(drop=True)
    # non-dominated plans
    nd = []
    for i, r in fr.iterrows():
        nd.append(not ((fr.distance_km <= r.distance_km) & (fr.loss_kg < r.loss_kg - 1e-9)).any())
    fr["non_dominated"] = nd
    P = fr[fr.non_dominated].copy()
    dn = (P.distance_km - P.distance_km.min()) / (P.distance_km.max() - P.distance_km.min())
    ln = (P.loss_kg - P.loss_kg.min()) / (P.loss_kg.max() - P.loss_kg.min())
    P["utopia_distance"] = np.sqrt(dn ** 2 + ln ** 2)
    P["km_per_kg_saved"] = (P.distance_km.diff() / -P.loss_kg.diff())
    fr = fr.merge(P[["distance_km", "loss_kg", "utopia_distance", "km_per_kg_saved"]], how="left",
                  on=["distance_km", "loss_kg"])
    fr.drop(columns="routes").to_csv(RES / "q3_tradeoff.csv", index=False)
    knee = P.loc[P.utopia_distance.idxmin()]
    # weighted-sum choice for different relative weights of the normalised loss
    choice = []
    for w in np.round(np.linspace(0.05, 0.95, 19), 2):
        s = (1 - w) * dn + w * ln
        j = s.idxmin()
        choice.append({"loss_weight": float(w), "eps": P.loc[j, "eps_label"], "helicopters": int(P.loc[j, "helicopters"]),
                       "distance_km": float(P.loc[j, "distance_km"]), "loss_kg": float(P.loc[j, "loss_kg"])})
    pd.DataFrame(choice).to_csv(RES / "q3_weight_choice.csv", index=False)
    # alternative loss laws with the same value at 300 min (plans not re-optimised)
    alt = []
    for pp in [2.0, 3.0, 4.0]:
        for _, r in P.iterrows():
            plan = to_nodes(r.routes, idx)
            alt.append({"exponent": pp, "eps": r.eps_label, "loss_kg": route_loss(D, plan, q, 0.12, pp)})
    pd.DataFrame(alt).to_csv(RES / "q3_loss_law.csv", index=False)
    # detail of the knee plan
    plan = to_nodes(knee.routes, idx)
    rt = route_table(D, plan, q, c, p, names)
    rt.to_csv(RES / "q3_knee_route_detail.csv", index=False)
    # figure: trade-off with knee
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    ax.semilogy(P.distance_km, P.loss_kg, "o-", c=CB[0], ms=4, label="non-dominated plans")
    dom = fr[~fr.non_dominated]
    if len(dom):
        ax.semilogy(dom.distance_km, dom.loss_kg, "x", c="#999999", label="dominated (time limit)")
    for _, r in P.iterrows():
        ax.annotate(f"{int(r.helicopters)}", (r.distance_km, r.loss_kg), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.plot(knee.distance_km, knee.loss_kg, "s", mfc="none", mec=CB[1], ms=9, label="compromise plan")
    ax.axhline(res["q3"]["direct_flight_lower_bound_kg"], c="#999999", lw=0.8, ls=":")
    ax.set_xlabel("total flight distance (km)")
    ax.set_ylabel("in-flight loss of supplies (kg)")
    ax.legend(fontsize=7)
    save(fig, "q3_tradeoff")
    return {"knee": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v)) for k, v in knee.items()},
            "front": P.drop(columns="routes").to_dict(orient="records")}, plan


def knee_route_figure(cities, base, D, plan):
    from run_all import draw_routes
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    draw_map(ax, cities, base)
    draw_routes(ax, cities, base, plan, D)
    ax.legend(loc="lower right", fontsize=7)
    save(fig, "q3_routes_compromise")


def main(part):
    cities, loss_table, q, c, p, k, base, D, res, idx = setup()
    names = list(cities["name_en"])
    dmin = res["q2"]["min_distance_km"]
    out_path = RES / "further_analysis.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    if part in ("all", "slow"):
        log("Problem 3: additional epsilon values")
        q3_extra_eps(D, q, c, p, names, dmin)
        log("Problem 2: range and payload")
        q2_range_payload(D, q)
    if part in ("all", "fast"):
        log("Problem 1: landscape around the Weber point")
        out["q1"], _, _ = q1_landscape(cities, k, base)
        log("Problem 2: route detail and heuristic check")
        out["q2"], _ = q2_checks(D, q, c, p, res, idx, names, k)
        log("Problem 3: trade-off analysis")
        out["q3"], plan = q3_tradeoff(D, q, c, p, res, idx, names)
        knee_route_figure(cities, base, D, plan)
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log("done")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
