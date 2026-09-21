"""Netze (kleines Netz, Stadtnetz, Zufallsnetz, Toronto Innenstadt), Paarwahl, Kennzahlen, Verteilungen, Experimente."""

import networkx as nx
import numpy as np
import pytest

import ch_algorithm as alg
import ch_bd as bd
import ch_constants as C
import ch_evaluation as ev
import ch_scenario as sc
from ch_graph import route_cost


def _components(g):
    G = nx.DiGraph()
    for u in range(g.n):
        for v in g.out(u):
            G.add_edge(u, int(v))
    return nx.number_strongly_connected_components(G)


# --- Netze -----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(3))
def test_city_is_connected_with_whole_number_costs_and_the_reverse_graph_matches(seed):
    net = sc.make_network("city", 12, 2.3, 1.0, 20, seed=seed)
    g = net.graph
    assert _components(g) == 1 and (g.weight >= 1).all() and np.array_equal(g.weight, np.rint(g.weight))
    assert np.array_equal(net.reverse.indices, g.indices) and np.array_equal(net.reverse.weight, g.weight)          # ungerichtet: derselbe Inhalt


def test_random_network_is_connected_has_the_requested_degree_and_is_deterministic():
    a, b, c = (sc.build_random(500, 4.0, s) for s in (1, 1, 2))
    assert _components(a) == 1 and abs(a.m / a.n - 4.0) < 0.05 and np.array_equal(a.indices, b.indices) and not np.array_equal(a.weight, c.weight)
    assert (a.weight >= 1).all() and (a.weight <= 9).all()


def test_toronto_downtown_data_is_the_expected_excerpt():
    net = sc.toronto_network()
    g = net.graph
    assert (g.n, g.m) == (10153, 26985) and g.directed and net.unit == "m"
    assert (g.weight >= 1).all() and np.array_equal(g.weight, np.rint(g.weight))
    assert _components(g) == 1                                                                   # größte stark zusammenhängende Komponente


def test_small_network_matches_its_definition_and_contraction_shows_shortcuts_and_witnesses():
    net = sc.small_network()
    g = net.graph
    assert g.n == 8 and g.m == 2 * len(sc.SMALL_LINKS) and not g.directed and net.fixed_pair == (0, 7)
    assert [g.names[i] for i in net.fixed_pair] == ["Altstadt", "Fabrik"]
    h = ev.build(net, "edge_difference")
    inserted = sum(1 for _, _, entries in h.log for e in entries if e[4])
    witnessed = sum(1 for _, _, entries in h.log for e in entries if not e[4])
    assert h.n_shortcuts == 6 and inserted == 6 and witnessed >= 1                              # mindestens eine Abkürzung UND ein Zeuge (Schritt-Ansicht zeigt beides)


# --- Paarwahl --------------------------------------------------------------------------------------------------------------------------------

def test_pick_pair_follows_the_distance_rank_and_is_deterministic():
    net = sc.make_network("city")
    pairs = {p: ev.pick_pair(net, p, 7) for p in (10, 50, 100)}
    assert pairs == {p: ev.pick_pair(net, p, 7) for p in (10, 50, 100)} and len({s for s, _ in pairs.values()}) == 1
    d = bd.dijkstra(net.graph, pairs[10][0]).dist
    assert d[pairs[10][1]] < d[pairs[50][1]] <= d[pairs[100][1]] and d[pairs[100][1]] == np.max(d)


def test_pick_pair_keeps_the_fixed_pair():
    small = sc.small_network()
    assert ev.pick_pair(small, 10, 1) == small.fixed_pair == ev.pick_pair(small, 99, 5)


# --- Kennzahlen ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["small", "city", "random"])
def test_analysis_invariants(key):
    net = sc.make_network(key)
    h = ev.build(net)
    s, t = ev.pick_pair(net)
    a = ev.analyse(net, h, s, t)
    m = a.metrics
    assert m["reachable"] and m["exact"] and m["cost_ch"] == m["cost_uni"] == m["cost_bi"] == pytest.approx(route_cost(net.graph, a.ch.route))
    assert m["settled_ch"] == m["settled_ch_f"] + m["settled_ch_b"] and m["gain_bi"] == pytest.approx(m["settled_bi"] / m["settled_ch"])
    assert a.ch.route[0] == s and a.ch.route[-1] == t and ev.verdict(a) in ("faster", "no_gain")
    assert m["n_shortcuts"] == h.n_shortcuts and m["packed_hops"] <= m["hops"] and 0 <= m["peak_rank"] < m["n"]


def test_analysis_of_an_unreachable_pair_reports_it():
    net = sc.make_network("random", nodes=200)
    from ch_graph import from_arcs
    g2 = from_arcs(4, [(0, 1, 1.0), (2, 3, 1.0)], np.zeros((4, 2)))
    net2 = type(net)(**{**net.__dict__, "graph": g2, "reverse": sc.reverse_graph(g2)})
    a = ev.analyse(net2, ev.build(net2), 0, 3)
    assert not a.metrics["reachable"] and ev.verdict(a) == "unreachable"


# --- Verteilung und Experimente ----------------------------------------------------------------------------------------------------------------

def test_pair_stats_fields_determinism_and_exactness():
    net = sc.make_network("city", 12)
    h = ev.build(net)
    a, b = ev.pair_stats(net, h, 30, 3), ev.pair_stats(net, h, 30, 3)
    assert a["n_pairs"] == 30 and np.array_equal(a["gain_bi"], b["gain_bi"]) and a["wrong"] == 0
    assert a["gain_bi_p10"] <= a["gain_bi_median"] <= a["gain_bi_p90"]
    assert 0.0 <= a["share_no_gain_bi"] <= 1.0


def test_order_comparison_rows_are_exact_and_random_is_worst():
    rows = ev.order_comparison(side=10, seeds=C.SWEEP_SEEDS[:2], pairs=8)
    assert [r["order"] for r in rows] == list(alg.ORDERS)
    ratio = {r["order"]: r["shortcut_ratio"] for r in rows}
    assert ratio["random"] > ratio["lazy_edge_difference"]


def test_city_sweep_rows():
    rows = ev.city_sweep("side", (8, 12), dict(side=10, reach=2.3, spread=1.0, blocked=20), pairs=6, seeds=C.SWEEP_SEEDS[:2])
    assert [r["value"] for r in rows] == [8, 12] and all(r["shortcut_ratio"] > 0 and r["gain_bi"] > 0 for r in rows)


def test_stale_weights_prove_the_old_hierarchy_wrong_and_a_rebuild_right():
    res = ev.stale_weights(fractions=(0.2,), pairs=15)
    row = res["rows"][0]
    assert row["share_wrong_cost"] > 0.3 and row["share_suboptimal"] > 0.1
    assert res["rebuild"] is None or res["rebuild"]["correct"] == res["rebuild"]["pairs"]
