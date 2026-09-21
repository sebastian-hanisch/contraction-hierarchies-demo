"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-21, Toleranzen fangen Rundung ab). Zähler, Abkürzungen, festgelegte Knoten und ganzzahlige Kosten sind plattformfest;
Laufzeiten stehen in der App nur als Messwerte und werden hier nie geprüft."""

import numpy as np
import pytest

import ch_algorithm as alg
import ch_constants as C
import ch_evaluation as ev
import ch_scenario as sc

BASE = dict(side=C.DEFAULT_SIDE, reach=C.DEFAULT_REACH, spread=C.DEFAULT_SPREAD, blocked=C.DEFAULT_BLOCKED)


@pytest.fixture(scope="module")
def toronto():
    net = sc.make_network("toronto")
    return net, ev.build(net)


def _analysis(key):
    p = C.PRESETS[{"small": "🔀 Kleines Netz", "city": "🏙️ Stadtnetz", "toronto": "🍁 Toronto Innenstadt", "random": "🕸️ Zufallsnetz"}[key]]
    net = sc.make_network(p["net"], p["side"], p["reach"], p["spread"], p["blocked"], p["nodes"], p["degree"], p["seed"])
    h = ev.build(net, p["order"], p["witness"], p["seed"])
    return net, h, ev.analyse(net, h, *ev.pick_pair(net, p["distance"], p["seed"]))


# --- Preset-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_small_preset_numbers():
    net, h, a = _analysis("small")
    assert h.n_shortcuts == 6 and a.metrics["exact"]
    assert "6 gerichtete Abkürzungen" in C.PRESET_HELP["🔀 Kleines Netz"]


def test_city_preset_numbers():
    _, _, a = _analysis("city")
    m = a.metrics
    assert (m["settled_ch"], m["settled_bi"], m["settled_uni"]) == (51, 126, 241)
    assert "51 Knoten" in C.PRESET_HELP["🏙️ Stadtnetz"] and "126" in C.PRESET_HELP["🏙️ Stadtnetz"] and "241" in C.PRESET_HELP["🏙️ Stadtnetz"]


def test_toronto_preset_numbers(toronto):
    net, h = toronto
    a = ev.analyse(net, h, *ev.pick_pair(net))
    m = a.metrics
    assert (m["settled_ch"], m["settled_bi"], m["settled_uni"]) == (176, 3547, 6093) and (m["n"], m["m"]) == (10153, 26985)
    assert h.n_shortcuts == 37340 and h.n_shortcuts / m["m"] == pytest.approx(1.4, abs=0.05)
    help_ = C.PRESET_HELP["🍁 Toronto Innenstadt"]
    assert all(s in help_ for s in ("176 Knoten", "3 547", "6 093", "37 340", "1.4 je Kante"))


def test_random_preset_numbers():
    net, h, a = _analysis("random")
    m = a.metrics
    assert h.n_shortcuts == 4293 and h.n_shortcuts / m["m"] == pytest.approx(1.8, abs=0.05)
    assert (m["settled_ch"], m["settled_bi"]) == (51, 41) and ev.verdict(a) == "no_gain"
    help_ = C.PRESET_HELP["🕸️ Zufallsnetz"]
    assert "4 293" in help_ and "1.8 je Kante" in help_ and "51 Knoten" in help_ and "(41)" in help_


# --- Sidebar-Hilfen: Sweeps über das Stadtnetz (Median-Gewinn gegen bidirektional, Mittel über 5 Datensätze) ------------------------------------

@pytest.mark.parametrize("parameter,values,expected", [
    ("side", (10, 20, 30), (0.8, 1.1, 1.4)),
    ("reach", (1.0, 1.5, 2.3, 3.2), (3.3, 1.9, 1.1, 0.9)),
    ("spread", (0.0, 1.0, 3.0), (1.0, 1.1, 1.3)),
    ("blocked", (0, 20, 60), (1.1, 1.1, 1.5)),
])
def test_city_slider_help_numbers(parameter, values, expected):
    rows = ev.city_sweep(parameter, values, BASE)
    assert [r["gain_bi"] for r in rows] == pytest.approx(list(expected), abs=0.06)


def test_random_net_degree_help_numbers():
    gains = []
    for deg in (2.5, 4.0, 6.0):
        g = []
        for sd in C.SWEEP_SEEDS:
            net = sc.make_network("random", nodes=600, degree=deg, seed=sd)
            g.append(ev.pair_stats(net, ev.build(net, seed=sd, trace=False), 20, sd)["gain_bi_median"])
        gains.append(float(np.mean(g)))
    assert gains == pytest.approx([1.2, 0.6, 0.4], abs=0.06) and gains[0] > 1 > gains[1] > gains[2]


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------------

def test_order_comparison_claims():
    rows = {r["order"]: r for r in ev.order_comparison()}
    assert rows["random"]["shortcut_ratio"] == pytest.approx(2.0, abs=0.1)                                # Sidebar-Hilfe: zufällig 2.0
    assert rows["lazy_edge_difference"]["shortcut_ratio"] == pytest.approx(0.69, abs=0.05) and rows["edge_difference"]["shortcut_ratio"] == pytest.approx(0.61, abs=0.05)
    assert 0.6 <= rows["edge_difference"]["shortcut_ratio"] and rows["lazy_edge_difference"]["shortcut_ratio"] <= 0.75                 # "0.6-0.7"
    assert rows["degree"]["shortcut_ratio"] == pytest.approx(0.4, abs=0.05)                              # README: "nur Grad" 0.4
    assert rows["degree"]["shortcut_ratio"] < rows["edge_difference"]["shortcut_ratio"]                 # "nur Grad" braucht sogar weniger Abkürzungen, aber die Abfrage legt mehr fest
    assert rows["degree"]["median_ch"] > rows["lazy_edge_difference"]["median_ch"] and rows["random"]["median_ch"] > rows["lazy_edge_difference"]["median_ch"]


def test_net_comparison_claims():
    rows = {r["net"]: r for r in ev.net_comparison()}
    assert rows["toronto"]["gain_bi"] == pytest.approx(21.3, abs=1.0) and rows["toronto"]["gain_bi"] > 10
    assert rows["city"]["gain_bi"] == pytest.approx(1.1, abs=0.1)
    assert rows["random"]["gain_bi"] == pytest.approx(0.5, abs=0.1) and rows["random"]["gain_bi"] < 1                # CH legt im Median MEHR fest als bidirektional
    assert rows["random"]["shortcut_ratio"] == pytest.approx(1.9, abs=0.1) and rows["toronto"]["shortcut_ratio"] == pytest.approx(1.4, abs=0.05)


def test_witness_limit_claims():
    rows = {r["witness"]: r for r in ev.witness_comparison()}
    assert [rows[k]["shortcut_ratio"] for k in (1, 5, 20, 200)] == pytest.approx([3.1, 2.2, 1.5, 1.4], abs=0.06)
    assert all(r["wrong"] == 0 for r in rows.values())                                                     # nie falsche Kosten, nur mehr Abkürzungen


def test_stale_weights_claims():
    res = ev.stale_weights()
    r = {round(x["fraction"], 2): x for x in res["rows"]}
    assert r[0.02]["share_wrong_cost"] == pytest.approx(0.78, abs=0.03) and r[0.02]["share_suboptimal"] == pytest.approx(0.60, abs=0.03)
    assert r[0.02]["excess_mean"] == pytest.approx(0.02, abs=0.005)
    assert r[0.2]["share_wrong_cost"] == pytest.approx(1.0) and r[0.2]["share_suboptimal"] == pytest.approx(0.95, abs=0.03)
    assert res["rebuild"]["correct"] == res["rebuild"]["pairs"] == 40                                      # neu vorrechnen macht alle Paare wieder exakt


# --- Grenzen der Aussagen ------------------------------------------------------------------------------------------------------------------------

def test_shortcuts_never_change_the_answer_only_the_effort(toronto):
    net, h = toronto
    ps = ev.pair_stats(net, h, 40, 9)
    assert ps["wrong"] == 0 and ps["median_ch"] < ps["median_bi"] < ps["median_uni"]
