"""Vorberechnung und Abfrage im Vergleich mit Dijkstra und der bidirektionalen Suche: Kennzahlen, Verteilung über Zufallspaare, Experimente (Ordnung, Netztyp, Zeugengrenze, veraltete Kosten), Urteil für die App."""

import time
from dataclasses import dataclass

import numpy as np

import ch_algorithm as alg
import ch_bd as bd
import ch_constants as C
from ch_graph import route_cost
from ch_scenario import make_network


@dataclass(frozen=True)
class Analysis:
    net: object
    h: alg.Hierarchy
    s: int
    t: int
    uni: bd.Unidirectional
    bi: bd.Bidirectional
    ch: alg.ChQuery
    metrics: dict
    seconds: dict


def pick_pair(net, distance_pct=C.DEFAULT_DISTANCE, seed=C.DEFAULT_SEED):
    """Start und Ziel: bei Netzen mit fester Aufgabe diese; sonst ein Start (Netze mit Karte: nahe bei 30 % Breite und 50 % Höhe, damit die Suchflächen nicht am Rand abgeschnitten werden; sonst zufällig)
    und als Ziel der Knoten, dessen Entfernung vom Start in der Rangfolge aller erreichbaren Knoten bei `distance_pct` Prozent liegt."""
    if net.fixed_pair:
        return net.fixed_pair
    g = net.graph
    rng = np.random.default_rng([int(seed), 808])
    if net.geometric:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        s = int(np.argmin(np.hypot(*(g.xy - (lo + (hi - lo) * np.array([0.3, 0.5]))).T)))
    else:
        s = int(rng.integers(0, g.n))
    d = bd.dijkstra(g, s).dist
    reachable = np.where(np.isfinite(d))[0]
    reachable = reachable[reachable != s]
    order = reachable[np.argsort(d[reachable], kind="stable")]
    t = int(order[min(len(order) - 1, int(round(distance_pct / 100.0 * (len(order) - 1))))])
    return s, t


def build(net, order=C.DEFAULT_ORDER, witness=C.DEFAULT_WITNESS, seed=C.DEFAULT_SEED, trace=None):
    """Vorberechnung für ein Netz; bei kleinen (benannten) Netzen mit Protokoll für die Schritt-Ansicht."""
    return alg.build_hierarchy(net.graph, order, witness_limit=witness, sim_limit=C.SIM_LIMIT_FOR.get(witness, min(witness, 50)), seed=seed, trace=bool(net.graph.names) if trace is None else trace)


def analyse(net, h, s, t):
    """Ein Paar: einseitiges Dijkstra, bidirektionale Suche (Stück 3 der Linie) und CH-Abfrage; die Kosten müssen übereinstimmen."""
    g, rg = net.graph, net.reverse
    t0 = time.perf_counter()
    uni = bd.dijkstra(g, s, t)
    t_uni = time.perf_counter() - t0
    t0 = time.perf_counter()
    bi = bd.bidirectional(g, rg, s, t)
    t_bi = time.perf_counter() - t0
    t0 = time.perf_counter()
    ch = alg.ch_query(h, s, t)
    t_ch = time.perf_counter() - t0
    reachable = uni.found >= 0
    ref = float(uni.dist[t]) if reachable else float("nan")
    su, sb, sc = len(uni.order), bi.settled, ch.settled
    saving = t_uni - t_ch
    metrics = {"reachable": reachable, "n": g.n, "m": g.m, "cost_uni": ref, "cost_bi": bi.cost, "cost_ch": ch.cost, "exact": bool(reachable and abs(ch.cost - ref) < 1e-9 and abs(bi.cost - ref) < 1e-9),
               "settled_uni": su, "settled_bi": sb, "settled_ch": sc, "settled_ch_f": len(ch.order_f), "settled_ch_b": len(ch.order_b),
               "gain_uni": su / max(sc, 1) if reachable else float("nan"), "gain_bi": sb / max(sc, 1) if reachable else float("nan"),
               "hops": len(ch.route) - 1 if ch.route else 0, "packed_hops": len(ch.packed) - 1 if ch.packed else 0, "shortcuts_used": ch.shortcuts_used,
               "peak_rank": int(h.rank[ch.peak]) if ch.peak >= 0 else -1, "n_shortcuts": h.n_shortcuts, "shortcut_ratio": h.n_shortcuts / max(g.m, 1), "n_arcs": g.m,
               "prep_seconds": h.seconds, "break_even": (h.seconds / saving) if saving > 0 else float("inf")}
    return Analysis(net, h, s, t, uni, bi, ch, metrics, {"uni": t_uni, "bi": t_bi, "ch": t_ch})


# --- Verteilung über zufällige Start-Ziel-Paare ------------------------------------------------------------------------------------------------

def pair_stats(net, h, pairs=C.PAIRS, seed=0):
    """Über zufällige erreichbare Paare: festgelegte Knoten von Dijkstra, bidirektionaler Suche und CH-Abfrage, Gewinn der CH-Abfrage gegen beide, Anteil der Paare, in denen sie NICHT weniger festlegt als die
    bidirektionale Suche, und die Zahl falscher Kosten (muss 0 sein)."""
    g, rg = net.graph, net.reverse
    rng = np.random.default_rng([int(seed), 1414])
    su, sb, sc, wrong, t_uni, t_ch = [], [], [], 0, [], []
    tries = 0
    while len(su) < pairs and tries < pairs * 20:
        tries += 1
        s, t = (int(x) for x in rng.integers(0, g.n, 2))
        if s == t:
            continue
        t0 = time.perf_counter()
        uni = bd.dijkstra(g, s, t)
        t_uni.append(time.perf_counter() - t0)
        if uni.found < 0:
            continue
        bi = bd.bidirectional(g, rg, s, t)
        t0 = time.perf_counter()
        q = alg.ch_query(h, s, t)
        t_ch.append(time.perf_counter() - t0)
        wrong += abs(q.cost - uni.dist[t]) > 1e-9
        su.append(len(uni.order))
        sb.append(bi.settled)
        sc.append(q.settled)
    su, sb, sc = np.array(su), np.array(sb), np.array(sc)
    nan = float("nan")
    if not len(su):
        return {"n_pairs": 0, "wrong": int(wrong), "settled_uni": su, "settled_bi": sb, "settled_ch": sc, "gain_bi": su, "median_uni": nan, "median_bi": nan, "median_ch": nan, "gain_bi_median": nan,
                "gain_bi_p10": nan, "gain_bi_p90": nan, "gain_uni_median": nan, "share_no_gain_bi": nan, "seconds_uni": nan, "seconds_ch": nan}
    gain_bi = sb / np.maximum(sc, 1)
    return {"n_pairs": len(su), "wrong": int(wrong), "settled_uni": su, "settled_bi": sb, "settled_ch": sc, "gain_bi": gain_bi, "median_uni": float(np.median(su)), "median_bi": float(np.median(sb)),
            "median_ch": float(np.median(sc)), "gain_bi_median": float(np.median(gain_bi)), "gain_bi_p10": float(np.quantile(gain_bi, 0.1)), "gain_bi_p90": float(np.quantile(gain_bi, 0.9)),
            "gain_uni_median": float(np.median(su / np.maximum(sc, 1))), "share_no_gain_bi": float((gain_bi <= 1.0).mean()), "seconds_uni": float(np.median(t_uni)), "seconds_ch": float(np.median(t_ch))}


# --- Experimente -------------------------------------------------------------------------------------------------------------------------------

def order_comparison(side=16, seeds=C.SWEEP_SEEDS, pairs=20):
    """Die Ordnung der Knoten: vier Ordnungen auf demselben erzeugten Stadtnetz (je Sweep-Datensatz): Abkürzungen im Verhältnis zu den Kanten, festgelegte Knoten je CH-Abfrage (Median), Vorberechnungszeit (Messwert)."""
    rows = []
    for order in alg.ORDERS:
        ratio, med, sec = [], [], []
        for sd in seeds:
            net = make_network("city", side, C.DEFAULT_REACH, C.DEFAULT_SPREAD, C.DEFAULT_BLOCKED, seed=sd)
            h = build(net, order, seed=sd, trace=False)
            ps = pair_stats(net, h, pairs, sd)
            assert ps["wrong"] == 0
            ratio.append(h.n_shortcuts / net.graph.m)
            med.append(ps["median_ch"])
            sec.append(h.seconds)
        rows.append({"order": order, "shortcut_ratio": float(np.mean(ratio)), "median_ch": float(np.mean(med)), "prep_seconds": float(np.mean(sec))})
    return rows


def net_comparison(pairs=30, seeds=C.SWEEP_SEEDS):
    """Wo eine Hierarchie da ist: Stadtnetz (Standard), Toronto Innenstadt und Zufallsnetz - Abkürzungen je Kante, Median der festgelegten Knoten (Dijkstra, bidirektional, CH) und Gewinn der CH-Abfrage gegen die
    bidirektionale Suche. Erzeugte Netze je Sweep-Datensatz gemittelt, Toronto einmal."""
    rows = []
    for key in ("city", "toronto", "random"):
        ratio, uni, bi, ch, gain, sec = [], [], [], [], [], []
        for net, sd in ([(make_network("toronto"), 0)] if key == "toronto" else [(make_network(key, seed=sd), sd) for sd in seeds]):
            h = build(net, seed=sd, trace=False)
            ps = pair_stats(net, h, pairs, sd)
            assert ps["wrong"] == 0
            ratio.append(h.n_shortcuts / net.graph.m)
            uni.append(ps["median_uni"])
            bi.append(ps["median_bi"])
            ch.append(ps["median_ch"])
            gain.append(ps["gain_bi_median"])
            sec.append(h.seconds)
        rows.append({"net": key, "shortcut_ratio": float(np.mean(ratio)), "median_uni": float(np.mean(uni)), "median_bi": float(np.mean(bi)), "median_ch": float(np.mean(ch)),
                     "gain_bi": float(np.mean(gain)), "prep_seconds": float(np.mean(sec))})
    return rows


def witness_comparison(pairs=30):
    """Die Zeugensuche begrenzen (Toronto Innenstadt): weniger Zeugen kosten Abkürzungen, nie Richtigkeit."""
    net = make_network("toronto")
    rows = []
    for limit in C.WITNESS_OPTIONS:
        h = build(net, witness=limit, trace=False)
        ps = pair_stats(net, h, pairs, 3)
        rows.append({"witness": limit, "shortcut_ratio": h.n_shortcuts / net.graph.m, "median_ch": ps["median_ch"], "wrong": ps["wrong"], "prep_seconds": h.seconds})
    return rows


def stale_weights(fractions=(0.02, 0.05, 0.2), factor=2.0, pairs=40):
    """Verkehr ändert sich (Toronto Innenstadt): ein Anteil der Kanten wird `factor`-mal so teuer. Die alte Hierarchie kennt die neuen Kosten nicht: ihre Route ist eine gültige, aber (fast immer) nicht mehr die kürzeste,
    und die berichteten Kosten sind die alten. Nach neuem Vorrechnen stimmt wieder alles."""
    net = make_network("toronto")
    g = net.graph
    h = build(net, trace=False)
    rng = np.random.default_rng(3)
    plist = []
    while len(plist) < pairs:
        s, t = (int(x) for x in rng.integers(0, g.n, 2))
        if s != t:
            plist.append((s, t))
    rows, rebuild = [], None
    for frac in fractions:
        g2 = alg.reweight(g, frac, factor, seed=5)
        wrong_cost = subopt = 0
        excess = []
        for s, t in plist:
            ref = bd.dijkstra(g2, s, t).dist[t]
            q = alg.ch_query(h, s, t)
            wrong_cost += abs(q.cost - ref) > 1e-9
            actual = route_cost(g2, q.route)
            if actual > ref + 1e-9:
                subopt += 1
                excess.append(actual / ref - 1)
        rows.append({"fraction": frac, "share_wrong_cost": wrong_cost / len(plist), "share_suboptimal": subopt / len(plist), "excess_mean": float(np.mean(excess)) if excess else 0.0})
        if rebuild is None and abs(frac - 0.05) < 1e-9:
            net2 = type(net)(**{**net.__dict__, "graph": g2})
            h2 = build(net2, trace=False)
            ok = sum(abs(alg.ch_query(h2, s, t).cost - bd.dijkstra(g2, s, t).dist[t]) < 1e-9 for s, t in plist)
            rebuild = {"seconds": h2.seconds, "correct": ok, "pairs": len(plist)}
    return {"rows": rows, "rebuild": rebuild, "prep_seconds": h.seconds}


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------------

def verdict(a):
    """Code für die App: unreachable / wrong / faster / no_gain (CH-Abfrage legt nicht weniger fest als die bidirektionale Suche)."""
    m = a.metrics
    if not m["reachable"]:
        return "unreachable"
    if not m["exact"]:
        return "wrong"
    return "faster" if m["settled_ch"] < m["settled_bi"] else "no_gain"


def city_sweep(parameter, values, base, pairs=20, seeds=C.SWEEP_SEEDS):
    """Ein Regler des Stadtnetzes durchgefahren (alle anderen wie in `base`: side, reach, spread, blocked): Gewinn der CH-Abfrage gegen die bidirektionale Suche (Median der Paare, Mittel über die Sweep-Datensätze)
    und Abkürzungen je Kante."""
    rows = []
    for v in values:
        kw = {**base, parameter: v}
        gain, ratio = [], []
        for sd in seeds:
            net = make_network("city", kw["side"], kw["reach"], kw["spread"], kw["blocked"], seed=sd)
            h = build(net, seed=sd, trace=False)
            ps = pair_stats(net, h, pairs, sd)
            gain.append(ps["gain_bi_median"])
            ratio.append(h.n_shortcuts / net.graph.m)
        rows.append({"value": v, "gain_bi": float(np.mean(gain)), "shortcut_ratio": float(np.mean(ratio))})
    return rows
