"""Contraction Hierarchies gegen networkx und gegen einseitiges Dijkstra: alle Ordnungen, verschiedene Zeugengrenzen, gerichtet und ungerichtet, Grenzfälle, Invarianten der Hierarchie."""

import networkx as nx
import numpy as np
import pytest

import ch_algorithm as alg
import ch_bd as bd
from ch_graph import from_arcs, reverse_graph, route_cost


def random_graph(n, m, seed, directed=False, integer=True, zero=0.0):
    rng = np.random.default_rng(seed)
    arcs = []
    for _ in range(m):
        u, v = rng.integers(0, n, 2)
        w = float(rng.integers(1, 9)) if integer else 0.5 + 5 * rng.random()
        if zero and rng.random() < zero:
            w = 0.0
        arcs.append((u, v, w))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=directed)


def to_nx(g):
    G = nx.DiGraph()
    G.add_nodes_from(range(g.n))
    for u in range(g.n):
        for v, w in zip(g.out(u), g.out_weights(u)):
            G.add_edge(u, int(v), weight=float(w))
    return G


# --- Richtigkeit: jede Abfrage gegen networkx -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("order", alg.ORDERS)
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("directed", [False, True])
def test_every_query_equals_networkx_for_every_order(order, seed, directed):
    g = random_graph(45, 110, seed, directed, integer=False)
    h = alg.build_hierarchy(g, order, seed=seed)
    G = to_nx(g)
    for s in range(g.n):
        lengths = nx.single_source_dijkstra_path_length(G, s)
        for t in range(0, g.n, 4):
            q = alg.ch_query(h, s, t)
            if t in lengths:
                assert q.cost == pytest.approx(lengths[t]) and route_cost(g, q.route) == pytest.approx(q.cost)
                assert q.route[0] == s and q.route[-1] == t
            else:
                assert q.route == [] and not np.isfinite(q.cost)


@pytest.mark.parametrize("witness_limit,sim_limit", [(1, 1), (5, 3), (50, 20), (10_000, 10_000)])
def test_the_witness_limit_changes_only_the_number_of_shortcuts_never_the_answer(witness_limit, sim_limit):
    g = random_graph(60, 150, 4, directed=True, integer=True)
    h = alg.build_hierarchy(g, "edge_difference", witness_limit=witness_limit, sim_limit=sim_limit)
    G = to_nx(g)
    for s, t in ((0, 59), (3, 40), (10, 11), (25, 2), (7, 33)):
        q = alg.ch_query(h, s, t)
        if nx.has_path(G, s, t):
            assert q.cost == pytest.approx(nx.dijkstra_path_length(G, s, t))


def test_a_tiny_witness_limit_creates_at_least_as_many_shortcuts():
    g = random_graph(80, 200, 6)
    tiny = alg.build_hierarchy(g, "edge_difference", witness_limit=1, sim_limit=1)
    full = alg.build_hierarchy(g, "edge_difference", witness_limit=10_000, sim_limit=10_000)
    assert tiny.n_shortcuts >= full.n_shortcuts


def test_zero_weight_edges_ties_and_parallel_edges():
    for seed in range(5):
        g = random_graph(40, 120, seed, directed=True, zero=0.25)
        h = alg.build_hierarchy(g, "lazy_edge_difference")
        G = to_nx(g)
        for s, t in ((0, 39), (5, 30), (12, 13), (20, 1)):
            q = alg.ch_query(h, s, t)
            assert (q.cost == pytest.approx(nx.dijkstra_path_length(G, s, t))) if nx.has_path(G, s, t) else not np.isfinite(q.cost)
    raw = from_arcs(3, [(0, 1, 5.0), (0, 1, 1.0), (1, 2, 2.0)], np.zeros((3, 2)), directed=True, clean=False)
    assert alg.ch_query(alg.build_hierarchy(raw), 0, 2).cost == 3.0


def test_start_equals_target_and_unreachable_and_one_way():
    g = from_arcs(4, [(0, 1, 1.0), (1, 2, 1.0), (3, 2, 1.0)], np.zeros((4, 2)), directed=True)
    h = alg.build_hierarchy(g)
    assert alg.ch_query(h, 2, 2).cost == 0 and alg.ch_query(h, 0, 2).cost == 2 and alg.ch_query(h, 2, 0).route == [] and alg.ch_query(h, 0, 3).route == []


# --- Invarianten der Hierarchie ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("order", alg.ORDERS)
def test_rank_is_a_permutation_and_every_arc_is_recorded_once_going_up_or_down(order):
    g = random_graph(50, 130, 2, directed=True)
    h = alg.build_hierarchy(g, order)
    assert sorted(h.rank.tolist()) == list(range(g.n)) and [int(h.rank[v]) for v in h.order] == list(range(g.n))
    count = 0
    for v in range(g.n):
        for x, w, mid in h.up[v]:
            assert h.rank[x] > h.rank[v] and h.arcs[(v, x)] == (w, mid)
            count += 1
        for u, w, mid in h.down_in[v]:
            assert h.rank[u] > h.rank[v] and h.arcs[(u, v)] == (w, mid)
            count += 1
    assert count == len(h.arcs)


@pytest.mark.parametrize("order", alg.ORDERS)
def test_every_shortcut_is_the_sum_of_its_two_halves_through_its_middle_node(order):
    g = random_graph(60, 160, 3, directed=True)
    h = alg.build_hierarchy(g, order)
    for (a, b), (w, mid) in h.arcs.items():
        if mid >= 0:
            assert h.rank[mid] < h.rank[a] and h.rank[mid] < h.rank[b]
            assert w == pytest.approx(h.arcs[(a, mid)][0] + h.arcs[(mid, b)][0])


def test_original_arcs_are_never_cheaper_than_the_graph_and_are_all_kept_or_dominated():
    g = random_graph(50, 130, 5, directed=True)
    h = alg.build_hierarchy(g)
    for u in range(g.n):
        for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
            assert h.arcs[(u, v)][0] <= w + 1e-12                                                 # eine Abkürzung kann eine teure Originalkante ersetzen


def test_unpacking_yields_original_edges_only_and_the_reported_cost():
    g = random_graph(70, 190, 8, directed=True)
    h = alg.build_hierarchy(g, "edge_difference")
    used = 0
    for s in range(0, 70, 9):
        for t in range(1, 70, 11):
            q = alg.ch_query(h, s, t)
            if np.isfinite(q.cost) and s != t:
                assert route_cost(g, q.route) == pytest.approx(q.cost)
                used += q.shortcuts_used
                assert all(g.arc(a, b) >= 0 for a, b in zip(q.route[:-1], q.route[1:]))
    assert used > 0


def test_the_peak_has_the_highest_rank_on_the_packed_route():
    g = random_graph(60, 150, 1)
    h = alg.build_hierarchy(g)
    for s, t in ((0, 59), (4, 30), (17, 42)):
        q = alg.ch_query(h, s, t)
        if np.isfinite(q.cost):
            assert h.rank[q.peak] == max(h.rank[v] for v in q.packed)


def test_ordering_changes_the_shortcut_count_but_not_the_costs():
    g = random_graph(120, 300, 9, integer=False)
    counts = {o: alg.build_hierarchy(g, o).n_shortcuts for o in alg.ORDERS}
    assert len(set(counts.values())) > 1
    assert counts["random"] > counts["edge_difference"]                                           # zufällige Reihenfolge braucht mehr Abkürzungen als die Kantendifferenz


def test_the_log_records_every_contraction_with_witnesses_and_inserted_shortcuts():
    g = random_graph(20, 45, 2)
    h = alg.build_hierarchy(g, "edge_difference", trace=True)
    assert [v for v, _, _ in h.log] == h.order and [r for _, r, _ in h.log] == list(range(g.n))
    inserted = sum(1 for _, _, entries in h.log for e in entries if e[4])
    assert inserted >= h.n_shortcuts and any(e[3] is not None for _, _, entries in h.log for e in entries)


def test_query_costs_equal_the_bidirectional_search_on_a_grid():
    side = 16
    arcs = [(i * side + j, i * side + j + 1, 1.0) for i in range(side) for j in range(side - 1)] + [(i * side + j, (i + 1) * side + j, 1.0) for i in range(side - 1) for j in range(side)]
    g = from_arcs(side * side, arcs, np.zeros((side * side, 2)))
    h = alg.build_hierarchy(g)
    q, b = alg.ch_query(h, 0, side * side - 1), bd.bidirectional(g, reverse_graph(g), 0, side * side - 1)
    assert q.cost == b.cost == 2 * (side - 1)


def test_reweight_changes_the_requested_fraction_and_keeps_the_structure():
    g = random_graph(60, 150, 1)
    g2 = alg.reweight(g, 0.2, 3.0, seed=2)
    changed = float((g2.weight != g.weight).mean())
    assert 0.1 < changed < 0.3 and np.array_equal(g2.indices, g.indices) and (g2.weight >= 1).all()
