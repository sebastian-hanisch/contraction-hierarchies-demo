"""Unabhängiges Orakel für Contraction Hierarchies: (1) Floyd-Warshall über alle Paare für jede CH-Abfrage inklusive Gleichstände, Nullkosten und unerreichbare Ziele;
(2) eine zweite Zusammenziehung mit vollständiger Zeugensuche (Floyd-Warshall im Restgraphen statt begrenztem Dijkstra) in der Reihenfolge der Demo: dieselben Hierarchiekanten mit denselben Kosten."""

import numpy as np
import pytest

import ch_algorithm as alg
from ch_graph import from_arcs, route_cost

INF = float("inf")


def floyd(n, M):
    M = M.copy()
    for k in range(n):
        M = np.minimum(M, M[:, k:k + 1] + M[k:k + 1, :])
    return M


def matrix(g):
    M = np.full((g.n, g.n), INF)
    for u in range(g.n):
        for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
            if u != v:
                M[u, v] = min(M[u, v], w)
    return M


def oracle_hierarchy_arcs(g, order):
    """Kanten der Hierarchie {(u, v): Kosten}, wenn man die Knoten in `order` zusammenzieht und eine Abkürzung genau dann einfügt, wenn der Restgraph ohne den Knoten keinen gleich kurzen Weg hat."""
    M, alive, recorded = matrix(g), set(range(g.n)), {}
    for v in order:
        outs = [w for w in alive if w != v and M[v, w] < INF]
        ins = [u for u in alive if u != v and M[u, v] < INF]
        recorded.update({(v, w): M[v, w] for w in outs})
        recorded.update({(u, v): M[u, v] for u in ins})
        rest = sorted(alive - {v})
        sub = floyd(len(rest), M[np.ix_(rest, rest)])
        idx = {x: i for i, x in enumerate(rest)}
        new = [(u, w, M[u, v] + M[v, w]) for u in ins for w in outs if u != w and sub[idx[u], idx[w]] > M[u, v] + M[v, w]]
        for u, w, via in new:
            M[u, w] = min(M[u, w], via)
        alive.discard(v)
    return recorded


def random_instance(rng, it):
    n, m = int(rng.integers(2, 11)), 0
    m = int(rng.integers(0, 3 * n))
    mode = it % 4
    arcs = []
    for _ in range(m):
        u, v = rng.integers(0, n, 2)
        w = [float(rng.integers(1, 4)), float(rng.integers(0, 3)), float(rng.integers(1, 10)), 0.5 + 5 * rng.random()][mode]
        arcs.append((u, v, w))
    return from_arcs(n, arcs, np.zeros((n, 2)), directed=bool(rng.integers(0, 2)))


@pytest.mark.parametrize("order", alg.ORDERS)
def test_queries_and_the_set_of_shortcuts_equal_an_independent_all_pairs_oracle(order):
    rng = np.random.default_rng(2024)
    for it in range(60):
        g = random_instance(rng, it)
        ref = floyd(g.n, matrix(g))
        np.fill_diagonal(ref, 0.0)
        h = alg.build_hierarchy(g, order, witness_limit=10 ** 6, sim_limit=50, seed=it)
        want = oracle_hierarchy_arcs(g, h.order)
        assert {k: v[0] for k, v in h.arcs.items()} == pytest.approx(want) and set(h.arcs) == set(want)
        for s in range(g.n):
            for t in range(g.n):
                q = alg.ch_query(h, s, t)
                if np.isfinite(ref[s, t]):
                    assert q.cost == pytest.approx(ref[s, t]) and route_cost(g, q.route) == pytest.approx(ref[s, t]) and (q.route[0], q.route[-1]) == (s, t)
                else:
                    assert q.route == [] and not np.isfinite(q.cost)


def test_a_tiny_witness_limit_still_gives_exact_costs_on_the_oracle():
    rng = np.random.default_rng(5)
    for it in range(40):
        g = random_instance(rng, it)
        ref = floyd(g.n, matrix(g))
        np.fill_diagonal(ref, 0.0)
        h = alg.build_hierarchy(g, "edge_difference", witness_limit=1, sim_limit=1)
        for s in range(g.n):
            for t in range(g.n):
                assert alg.ch_query(h, s, t).cost == pytest.approx(ref[s, t])
