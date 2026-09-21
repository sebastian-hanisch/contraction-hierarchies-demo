"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, alle Ordnungen und Zeugengrenzen, Randgrößen, Schritt-Zustand, ausgeblendete Regler, Permalink, Experimente auf Abruf, Schlüssel."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ch_constants as C
from ch_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
# Urteil je Preset: "success" = CH-Abfrage legt weniger fest als die bidirektionale Suche, "info" = kein Gewinn (gemessen, siehe test_claims)
EXPECTED_KIND = {"🔀 Kleines Netz": "info", "🏙️ Stadtnetz": "success", "🍁 Toronto Innenstadt": "success", "🕸️ Zufallsnetz": "info"}


def _run(setup=None, timeout=600, net=None):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    if net:
        at.query_params["net"] = net
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def _labels(at):
    return {w.label for w in list(at.sidebar.slider) + list(at.sidebar.selectbox) + list(at.sidebar.number_input)}


def test_default_renders_without_exception():
    at = _run()
    assert any("Die Abfrage in Aktion" in m.value for m in at.markdown)
    assert len(at.warning) == 0 and len(at.error) == 0


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders_with_its_verdict_kind(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    kind = EXPECTED_KIND[name]
    assert not at.warning
    assert (len(at.success) == 1 and len(at.info) == 0) if kind == "success" else (len(at.info) >= 1 and len(at.success) == 0)


@pytest.mark.parametrize("order", ["lazy_edge_difference", "edge_difference", "degree", "random"])
@pytest.mark.parametrize("witness", list(C.WITNESS_OPTIONS))
def test_every_order_and_witness_limit_renders_on_the_small_net(order, witness):
    def setup(at):
        at.session_state["order_select"] = order
        at.session_state["witness_select"] = witness
    at = _run(setup)
    assert not at.warning and len(at.success) + len(at.info) == 1


def test_random_order_is_only_offered_for_small_networks():
    def order_options(at):
        return [o for o in at.sidebar.selectbox(key="order_select").options]
    small = _run()
    assert len(order_options(small)) == len(C.ORDER_LABELS)
    toronto = _run(net="toronto")
    assert len(order_options(toronto)) == len(C.ORDER_LABELS) - 1
    # ein gewählter "random" wird beim Wechsel auf ein großes Netz auf den Standard zurückgesetzt
    at = _run(lambda a: a.session_state.__setitem__("order_select", "random"))
    at.session_state["net_select"] = "toronto"
    at.run()
    assert not at.exception and at.selectbox(key="order_select").value == C.DEFAULT_ORDER


def test_extreme_settings_render():
    def small(at):
        at.session_state["net_select"] = "city"
        at.session_state["side_slider"] = C.SIDE_MIN
        at.session_state["distance_slider"] = C.DISTANCE_MIN
    def big(at):
        at.session_state["net_select"] = "random"
        at.session_state["nodes_slider"] = C.NODES_MAX
        at.session_state["degree_slider"] = C.DEGREE_MAX
        at.session_state["distance_slider"] = C.DISTANCE_MAX
    for setup in (small, big):
        at = _run(setup)
        assert at.slider(key="ch_step").value == at.slider(key="ch_step").max


def test_hidden_controls_follow_the_net():
    def labels_for(net):
        return _labels(_run(net=net))
    small, city, rnd, tor = (labels_for(n) for n in ("small", "city", "random", "toronto"))
    assert small == {"Netz", "Ordnung der Knoten", "Zeugensuche (Knoten je Suche)"}                      # feste Aufgabe: kein Abstand, kein Seed
    assert {"Kreuzungen je Seite", "Reichweite der Straßen [Blocklängen]", "Streuung der Kosten", "Gesperrte Straßen [%]", "Entfernung Start–Ziel [%]", "Zufalls-Seed"} <= city and "Knoten" not in city
    assert {"Knoten", "Mittlerer Grad", "Entfernung Start–Ziel [%]", "Zufalls-Seed"} <= rnd and "Kreuzungen je Seite" not in rnd
    assert tor == {"Netz", "Entfernung Start–Ziel [%]", "Ordnung der Knoten", "Zeugensuche (Knoten je Suche)"}


def test_hidden_slider_values_come_back_when_the_net_is_shown_again():
    # Die erste Sicht muss das Netz mit dem Regler sein: AppTest verliert den Wert, wenn der Regler zuerst ausgeblendet war (im echten Browser bleibt er erhalten).
    at = _run(net="city")
    at.session_state["spread_slider"] = 2.5
    at.run()
    at.session_state["net_select"] = "toronto"
    at.run()
    at.session_state["net_select"] = "city"
    at.run()
    assert not at.exception and at.slider(key="spread_slider").value == 2.5


def test_step_slider_returns_to_the_last_step_when_anything_changes():
    at = _run(net="city")
    at.slider(key="ch_step").set_value(5)
    at.run()
    assert at.slider(key="ch_step").value == 5
    at.session_state["order_select"] = "degree"
    at.run()
    assert not at.exception and at.slider(key="ch_step").value == at.slider(key="ch_step").max


def test_every_query_step_and_every_contraction_step_of_the_small_net_render():
    at = _run()
    for k in range(0, int(at.slider(key="ch_step").max) + 1):
        at.slider(key="ch_step").set_value(k)
        at.run()
        assert not at.exception, k
    for k in range(0, int(at.slider(key="ch_contract").max) + 1):
        at.slider(key="ch_contract").set_value(k)
        at.run()
        assert not at.exception, k


def test_display_checkboxes_and_distance_slider_change_the_view():
    at = _run(net="city")
    at.checkbox(key="show_uni").set_value(False)
    at.checkbox(key="show_bi").set_value(False)
    at.run()
    assert not at.exception
    at.slider(key="distance_slider").set_value(20)
    at.run()
    assert not at.exception and at.slider(key="ch_step").value == at.slider(key="ch_step").max


def test_permalink_parameters_select_the_net_and_are_clamped():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "random"
    at.query_params["nodes"] = "999999"
    at.query_params["order"] = "never"
    at.query_params["wit"] = "5"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == "random"
    assert at.slider(key="nodes_slider").value == C.NODES_MAX and at.selectbox(key="order_select").value == C.DEFAULT_ORDER and at.selectbox(key="witness_select").value == 5


def test_unknown_net_in_the_permalink_falls_back_to_the_default():
    at = AppTest.from_file(str(APP), default_timeout=600)
    at.query_params["net"] = "ring"
    at.run()
    assert not at.exception and at.selectbox(key="net_select").value == C.DEFAULT_NET


def test_experiments_run_on_demand():
    at = _run()
    assert not any("über 30 zufällige Paare" in c.value for c in at.caption)
    for key in ("netcmp_start", "order_start", "witness_start", "stale_start"):
        at.button(key=key).click()
        at.run()
        assert not at.exception, (key, [e.value for e in at.exception])
    text = " ".join(c.value for c in at.caption)
    for needle in ("Median der festgelegten Knoten über 30 zufällige Paare", "Die **zufällige** Reihenfolge braucht", "Eine Zeugensuche darf abbrechen", "Die Hierarchie kennt nur die alten Kosten"):
        assert needle in text, needle


def _calls(src, name):
    """Der Text jedes Aufrufs `name(...)` einschließlich verschachtelter Klammern."""
    out = []
    for m in re.finditer(re.escape(name) + r"\(", src):
        depth, i = 1, m.end()
        while depth:
            depth += {"(": 1, ")": -1}.get(src[i], 0)
            i += 1
        out.append(src[m.start():i])
    return out


def test_every_plotly_chart_has_an_explicit_unique_key_and_axes_are_locked():
    calls = _calls(APP.read_text(encoding="utf-8"), "plotly_chart")
    keys = [re.search(r'key=f?"([a-z_]+?)(?:_\{\w+\})?"', c).group(1) for c in calls]
    assert len(calls) == 8 and len(set(keys)) == 8, keys
    viz = (ROOT / "ch_visualization.py").read_text(encoding="utf-8")
    assert "fixedrange=True" in viz and viz.count("_base(fig") >= 6


def test_app_text_has_no_links_to_repository_files():
    assert not re.search(r"\]\(\w+\.py\)", APP.read_text(encoding="utf-8"))


def test_play_runs_through_all_frames_without_duplicate_chart_keys():
    """Beim Abspielen entstehen in einem Lauf mehrere Diagramme mit demselben Namen - die Schlüssel tragen deshalb den Schritt (Regression: StreamlitDuplicateElementKey bei mehr als einem Bild)."""
    at = _run()
    [b for b in at.button if b.label == "▶️ Abspielen"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
