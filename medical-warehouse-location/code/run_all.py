"""Reproduce every number, table and figure of the report.

    python code/run_all.py            # problems 1-4, about 1 hour on a laptop
    python code/run_all.py q123       # problems 1-3 only
    python code/run_all.py q4         # problem 4 only

Outputs go to results/ (JSON/CSV) and results/figures/ (PDF + PNG).
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import model
from model import (FLEET, RANGE, SPEED, direct_loss, distance_matrix, fit_loss_curve,
                   genetic_algorithm, haversine, load_data, route_distance, route_lengths,
                   route_loss, served_at_base, solve_routing, total_distance, weber_point)

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.fontsize": 8,
                     "savefig.bbox": "tight", "figure.dpi": 150})
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442",
      "#999999", "#882255"]

# English names for the 21 prefecture-level units (attachment order)
EN = {"成都市": "Chengdu", "自贡市": "Zigong", "攀枝花市": "Panzhihua", "泸州市": "Luzhou",
      "德阳市": "Deyang", "绵阳市": "Mianyang", "广元市": "Guangyuan", "遂宁市": "Suining",
      "内江市": "Neijiang", "乐山市": "Leshan", "南充市": "Nanchong", "眉山市": "Meishan",
      "宜宾市": "Yibin", "广安市": "Guang'an", "达州市": "Dazhou", "雅安市": "Ya'an",
      "巴中市": "Bazhong", "资阳市": "Ziyang", "阿坝州": "Aba", "甘孜州": "Garze", "凉山州": "Liangshan"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tangent_gradient(point, cities, weights=None, h=1e-3):
    """Gradient of the (weighted) distance sum in local east/north km coordinates."""
    lon, lat = point
    kx = 111.195 * np.cos(np.radians(lat))  # km per degree of longitude
    ky = 111.195
    f = lambda p: total_distance(p, cities, weights)
    gx = (f((lon + h / kx, lat)) - f((lon - h / kx, lat))) / (2 * h)
    gy = (f((lon, lat + h / ky)) - f((lon, lat - h / ky))) / (2 * h)
    return np.array([gx, gy])


def vertex_condition(k, cities, weights=None):
    """Kuhn's optimality test for a Weber point located at demand point k:
    the point is optimal iff ||sum_{j != k} w_j e_kj|| <= w_k, where e_kj is the
    unit tangent vector pointing away from j."""
    others = cities.drop(index=k)
    w = None if weights is None else np.delete(weights, k)
    g = tangent_gradient(cities.loc[k, ["lon", "lat"]].to_numpy(float), others, w)
    wk = 1.0 if weights is None else float(weights[k])
    return float(np.linalg.norm(g)), wk


def main():
    t0 = time.time()
    cities, loss_table = load_data()
    cities["name_en"] = cities["city"].map(EN)
    q = cities["demand"].to_numpy(float)
    out = {"n_cities": int(len(cities)), "total_demand_kg": float(q.sum())}

    # ------------------------------------------------------------ problem 1
    log("Problem 1: spherical Weber point")
    f = lambda p: total_distance(p, cities)
    centroid = cities[["lon", "lat"]].mean().to_numpy()
    weber, f_weber = weber_point(cities)
    ga_runs = []
    histories = []
    for seed in range(20):
        x, fx, hist = genetic_algorithm(f, seed)
        ga_runs.append({"seed": seed, "lon": x[0], "lat": x[1], "total_km": fx,
                        "offset_km": float(haversine(x[0], x[1], weber[0], weber[1]))})
        histories.append(hist)
    ga = pd.DataFrame(ga_runs)
    ga.to_csv(RES / "q1_ga_runs.csv", index=False)
    nearest = int(np.argmin(haversine(weber[0], weber[1], cities["lon"], cities["lat"])))
    gnorm, wk = vertex_condition(nearest, cities)
    out["q1"] = {
        "centroid": {"lon": centroid[0], "lat": centroid[1], "total_km": f(centroid)},
        "weber": {"lon": weber[0], "lat": weber[1], "total_km": f_weber},
        "weber_coincides_with": cities.loc[nearest, "name_en"],
        "weber_to_city_km": float(haversine(weber[0], weber[1], cities.loc[nearest, "lon"], cities.loc[nearest, "lat"])),
        "vertex_condition_norm": gnorm,
        "vertex_condition_weight": wk,
        "centroid_excess_km": f(centroid) - f_weber,
        "centroid_to_weber_km": float(haversine(*centroid, *weber)),
        "ga_runs": len(ga),
        "ga_total_mean": float(ga.total_km.mean()),
        "ga_total_std": float(ga.total_km.std(ddof=1)),
        "ga_total_max": float(ga.total_km.max()),
        "ga_offset_mean_km": float(ga.offset_km.mean()),
        "ga_offset_max_km": float(ga.offset_km.max()),
    }
    # base = hospital site at the Weber point
    base = cities.loc[nearest, ["lon", "lat"]].to_numpy(float)
    D = distance_matrix(base, cities)

    # ------------------------------------------------------------ loss curve
    c, p, maxdev = fit_loss_curve(loss_table)
    out["loss_fit"] = {"c": c, "p": p, "max_abs_dev": maxdev}

    # ------------------------------------------------------------ problem 2
    log("Problem 2: routing MILP for k = 1..10")
    model.RANGE = 5000.0  # longer than any tour: range constraint inactive -> TSP
    tsp = solve_routing(D, q, k=1)
    model.RANGE = RANGE
    tsp_lb = tsp["bound"] if tsp["bound"] is not None else tsp["distance"]
    rows = []
    q2 = {}
    for k in range(1, FLEET + 1):
        if k == 1 and tsp_lb > RANGE:
            rows.append({"k": 1, "feasible": False, "status": "TSP lower bound exceeds range",
                         "distance_km": None, "longest_route_km": None})
            continue
        r = solve_routing(D, q, k=k)
        feasible = r["routes"] is not None
        rows.append({"k": k, "feasible": feasible, "status": r["status"], "mip_gap": r.get("mip_gap"),
                     "distance_km": r.get("distance"),
                     "longest_route_km": max(route_lengths(D, r["routes"])) if feasible else None})
        if feasible:
            q2[k] = r
        log(f"  k={k}: {rows[-1]}")
    q2tab = pd.DataFrame(rows)
    q2tab.to_csv(RES / "q2_fleet_size.csv", index=False)
    dbest = q2tab.distance_km.min()
    kbest = int(q2tab.loc[q2tab.distance_km <= dbest + 1e-6, "k"].min())  # fewest helicopters at the optimum
    best2 = q2[kbest]
    dmin = best2["distance"]
    flying = served_at_base(D, best2["routes"])
    out["q2"] = {
        "tsp_km": tsp["distance"], "tsp_lower_bound_km": tsp_lb, "tsp_status": tsp["status"],
        "k1_infeasible_because_tsp_exceeds_range": tsp_lb > RANGE,
        "min_distance_km": dmin, "k_at_min": kbest, "helicopters_flying": len(flying),
        "routes": [[cities.loc[j - 1, "name_en"] for j in r] for r in flying],
        "route_km": [route_distance(D, [r]) for r in flying],
        "route_hours": [route_distance(D, [r]) / SPEED for r in flying],
        "route_load_kg": [float(sum(q[j - 1] for j in r)) for r in flying],
        "loss_kg": route_loss(D, flying, q, c, p),
        "mip_gap": best2.get("mip_gap"),
    }

    # ------------------------------------------------------------ problem 3
    log("Problem 3: distance-loss trade-off (epsilon constraint)")
    front = []
    for eps in [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]:
        r = solve_routing(D, q, loss=(c, p), distance_cap=dmin * (1 + eps) + 1e-6)
        if r["routes"] is None:  # no feasible plan found within the time limit
            log(f"  eps={eps}: no plan found ({r['message']})")
            continue
        fly = [x for x in r["routes"] if route_distance(D, [x]) > 1e-9]
        front.append({"eps": eps, "distance_km": r["distance"], "loss_kg": route_loss(D, r["routes"], q, c, p),
                      "helicopters": len(fly), "status": r["status"], "mip_gap": r.get("mip_gap"),
                      "max_arrival_min": max(sum(D[a, b] for a, b in zip([0] + x[:-1], x)) for x in fly) / SPEED * 60,
                      "routes": [[cities.loc[j - 1, "name_en"] for j in x] for x in fly]})
        log(f"  eps={eps}: D={front[-1]['distance_km']:.1f} loss={front[-1]['loss_kg']:.2f} k={len(fly)}")
    rfree = solve_routing(D, q, loss=(c, p))
    fly = [x for x in rfree["routes"] if route_distance(D, [x]) > 1e-9]
    front.append({"eps": None, "distance_km": rfree["distance"], "loss_kg": route_loss(D, rfree["routes"], q, c, p),
                  "helicopters": len(fly), "status": rfree["status"], "mip_gap": rfree.get("mip_gap"),
                  "max_arrival_min": max(sum(D[a, b] for a, b in zip([0] + x[:-1], x)) for x in fly) / SPEED * 60,
                  "routes": [[cities.loc[j - 1, "name_en"] for j in x] for x in fly]})
    pd.DataFrame(front).drop(columns="routes").to_csv(RES / "q3_pareto.csv", index=False)
    out["q3"] = {"front": front, "direct_flight_lower_bound_kg": direct_loss(D, q, c, p)}

    # sensitivity: speed range given in the problem (200-400 km/h)
    sens = []
    for v in [200.0, 255.0, 300.0, 400.0]:
        model.SPEED = v
        r_d = solve_routing(D, q, k=kbest)
        r_l = solve_routing(D, q, loss=(c, p))
        sens.append({"speed_kmh": v, "q2_loss_kg": route_loss(D, served_at_base(D, r_d["routes"]), q, c, p),
                     "minloss_kg": route_loss(D, r_l["routes"], q, c, p), "minloss_distance_km": r_l["distance"]})
    model.SPEED = SPEED
    pd.DataFrame(sens).to_csv(RES / "q3_speed_sensitivity.csv", index=False)
    out["q3"]["speed_sensitivity"] = sens

    out["runtime_s"] = time.time() - t0
    (RES / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    log(f"done in {out['runtime_s']:.0f} s")

    # ------------------------------------------------------------ figures
    figures(cities, base, centroid, histories, D, best2, front, rfree, loss_table, c, p)


def draw_map(ax, cities, base):
    ax.scatter(cities.lon, cities.lat, s=14, c="#555555", zorder=3)
    for _, r in cities.iterrows():
        ax.annotate(r["name_en"], (r.lon, r.lat), xytext=(3, 2), textcoords="offset points", fontsize=7)
    ax.plot(*base, marker="*", ms=13, c=CB[1], zorder=4, ls="none", label="base")
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.set_aspect(1 / np.cos(np.radians(30)))


def draw_routes(ax, cities, base, routes, D):
    pts = np.vstack([base, cities[["lon", "lat"]].to_numpy()])
    styles = ["-", "--", "-.", ":"]
    for t, r in enumerate([r for r in routes if route_distance(D, [r]) > 1e-9]):
        path = [0] + list(r) + [0]
        ax.plot(pts[path, 0], pts[path, 1], styles[t % 4], c=CB[t % len(CB)], lw=1.2,
                label=f"helicopter {t + 1}: {route_distance(D, [r]):.0f} km")


def figures(cities, base, centroid, histories, D, best2, front, rfree, loss_table, c, p):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    draw_map(ax, cities, base)
    ax.plot(*centroid, marker="x", ms=8, c=CB[0], ls="none", label="coordinate mean")
    ax.legend(loc="lower right")
    save(fig, "q1_location")

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    H = np.array(histories)
    best = H.min()
    ax.semilogy(np.arange(H.shape[1]), H.mean(0) - best + 1e-3, c=CB[0], label="mean over 20 seeds")
    ax.semilogy(np.arange(H.shape[1]), H.max(0) - best + 1e-3, "--", c=CB[1], label="worst seed")
    ax.set_xlabel("generation")
    ax.set_ylabel("best total distance − optimum (km)")
    ax.legend()
    save(fig, "q1_ga_convergence")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    draw_map(ax, cities, base)
    draw_routes(ax, cities, base, best2["routes"], D)
    ax.legend(loc="lower right")
    save(fig, "q2_routes")

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    draw_map(ax, cities, base)
    draw_routes(ax, cities, base, rfree["routes"], D)
    ax.legend(loc="lower right", fontsize=6)
    save(fig, "q3_routes_minloss")

    t = loss_table.minutes.to_numpy(float)
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(t, 100 * loss_table.rate, "o", ms=3, c=CB[0], label="attachment 3")
    tt = np.linspace(0, 480, 200)
    ax.plot(tt, 100 * c * (tt / 300) ** p, "-", c=CB[1], label=f"fit $c(t/300)^p$, p = {p:.3f}")
    ax.axvline(300, c="#999999", lw=0.8)
    ax.set_xlabel("time on board (min)")
    ax.set_ylabel("loss rate (%)")
    ax.legend()
    save(fig, "q3_loss_curve")



def problem4(time_limit=60):
    """Re-evaluate the base location on candidate sites (hospital sites, the
    demand-weighted Weber point and a 0.25-degree grid around the Weber point).
    For each site: shortest total route (any number of helicopters) and the
    minimum-loss plan with up to 10 helicopters; each MILP is limited to
    `time_limit` seconds and its status/gap is recorded."""
    t0 = time.time()
    cities, loss_table = load_data()
    cities["name_en"] = cities["city"].map(EN)
    q = cities["demand"].to_numpy(float)
    c, p, _ = fit_loss_curve(loss_table)
    base_w, _ = weber_point(cities)
    nearest = int(np.argmin(haversine(base_w[0], base_w[1], cities["lon"], cities["lat"])))
    base = cities.loc[nearest, ["lon", "lat"]].to_numpy(float)
    log("Problem 4: re-evaluating the base location on candidate sites")
    wweber, f_ww = weber_point(cities, weights=q)
    cand = [("Weber point (Ziyang)", base[0], base[1]), ("demand-weighted Weber", wweber[0], wweber[1])]
    for i, row in cities.iterrows():
        if i != nearest:
            cand.append((row["name_en"], row["lon"], row["lat"]))
    for lon in np.arange(104.0, 105.01, 0.25):
        for lat in np.arange(29.75, 30.76, 0.25):
            cand.append((f"grid {lon:.2f}E {lat:.2f}N", lon, lat))
    rows = []
    for name, lon, lat in cand:
        Dc = distance_matrix((lon, lat), cities)
        rd = solve_routing(Dc, q, max_k=FLEET, time_limit=time_limit)
        rl = solve_routing(Dc, q, loss=(c, p), time_limit=time_limit)
        rows.append({"site": name, "lon": lon, "lat": lat,
                     "sum_distance_km": total_distance((lon, lat), cities),
                     "weighted_distance_tkm": total_distance((lon, lat), cities, q) / 1000,
                     "min_route_km": rd["distance"] if rd["routes"] else np.nan,
                     "min_route_status": rd["status"], "min_route_gap": rd.get("mip_gap"),
                     "min_loss_kg": route_loss(Dc, rl["routes"], q, c, p) if rl["routes"] else np.nan,
                     "min_loss_status": rl["status"],
                     "min_loss_route_km": rl["distance"] if rl["routes"] else np.nan})
        log(f"  {name}: {rows[-1]['min_route_km']:.1f} km, min loss {rows[-1]['min_loss_kg']:.2f} kg")
    q4 = pd.DataFrame(rows)
    q4.to_csv(RES / "q4_candidates.csv", index=False)
    opt = q4[q4.min_route_status == 0]
    res = {
        "weighted_weber": {"lon": wweber[0], "lat": wweber[1], "weighted_tkm": f_ww / 1000},
        "best_by_route_distance": opt.loc[opt.min_route_km.idxmin()].to_dict(),
        "best_by_min_loss": q4.loc[q4.min_loss_kg.idxmin()].to_dict(),
        "weber_row": q4.iloc[0].to_dict(),
        "n_candidates": len(q4), "n_route_optimal": int((q4.min_route_status == 0).sum()),
        "n_loss_optimal": int((q4.min_loss_status == 0).sum()), "runtime_s": time.time() - t0,
    }
    (RES / "results_q4.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, col, lab in [(axes[0], "min_route_km", "a) shortest total route (km)"),
                         (axes[1], "min_loss_kg", "b) minimum loss with 10 helicopters (kg)")]:
        sc = ax.scatter(q4.lon, q4.lat, c=q4[col], cmap="viridis", s=28)
        ax.plot(*base, marker="*", ms=12, c=CB[1], ls="none")
        fig.colorbar(sc, ax=ax, shrink=0.85)
        ax.set_xlabel("longitude (°E)")
        ax.set_ylabel("latitude (°N)")
        ax.set_title(lab, loc="left")
    save(fig, "q4_candidates")
    log(f"problem 4 done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    import sys
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("all", "q123"):
        main()
    if part in ("all", "q4"):
        problem4()
