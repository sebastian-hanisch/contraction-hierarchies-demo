"""Contraction Hierarchies - erst vorrechnen, dann blitzschnell fragen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - Contraction Hierarchies - und lässt stattdessen das Beispiel wachsen.
Viertes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe, Fortsetzung der Demo zur bidirektionalen Suche: dort begann jede Anfrage von vorn - hier wird einmal vorgerechnet, und die Abfrage sucht nur noch aufwärts.
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import ch_constants as C
import ch_evaluation as ev
from ch_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    seed_widget,
    sync_query_params,
)
from ch_scenario import make_network
from ch_visualization import build_added, build_contraction, build_gain_hist, build_net_comparison, build_order, build_query, build_stale, build_witness

st.set_page_config(page_title="Contraction Hierarchies – Sebastian Hanisch", layout="wide")


def _pct(x):
    return "–" if x is None or np.isnan(x) else f"{x:.0%}"


def _cost(net, x):
    return f"{x:,.0f} {net.unit}".replace(",", ".")


def _num(x):
    return f"{x:,}".replace(",", ".")


@st.cache_resource(show_spinner=False, max_entries=8)
def _network(params):
    return make_network(*params)


@st.cache_resource(show_spinner=False, max_entries=8)
def _hierarchy(params, order, witness):
    return ev.build(_network(params), order, witness, params[-1])


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params, order, witness, distance):
    net = _network(params)
    s, t = ev.pick_pair(net, distance, params[-1])
    return ev.analyse(net, _hierarchy(params, order, witness), s, t)


@st.cache_data(show_spinner=False)
def _pair_stats(params, order, witness, pairs):
    return ev.pair_stats(_network(params), _hierarchy(params, order, witness), pairs, params[-1])


@st.cache_data(show_spinner=False)
def _order_comparison():
    return ev.order_comparison()


@st.cache_data(show_spinner=False)
def _net_comparison():
    return ev.net_comparison()


@st.cache_data(show_spinner=False)
def _witness_comparison():
    return ev.witness_comparison()


@st.cache_data(show_spinner=False)
def _stale_weights():
    return ev.stale_weights()


st.title("🛣️ Contraction Hierarchies – erst vorrechnen, dann blitzschnell fragen")
st.markdown(
    """
Die bidirektionale Suche legt für ein Paar im Toronto-Netz mehrere tausend Knoten fest, und die nächste Anfrage beginnt wieder bei null. **Contraction Hierarchies (CH)** kehren das um: einmal **vorrechnen**, danach ist jede Anfrage winzig.
Die Vorberechnung ordnet die Knoten nach Wichtigkeit und **zieht** sie von unten nach oben **zusammen**: wird ein Knoten entfernt, ersetzt eine **Abkürzung** die Wege, die über ihn führten - es sei denn, ein **Zeuge** (ein anderer, gleich kurzer Weg) macht sie überflüssig.
Die Abfrage sucht dann von beiden Enden nur noch **aufwärts** in der Hierarchie und trifft sich an der wichtigsten Stelle der Route. Der Preis: Vorrechnen, feste Kosten - und ein Netz, das überhaupt eine Hierarchie hat.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - viertes Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe, Fortsetzung der Demo zur bidirektionalen Suche - **ein** Verfahren an einem wachsenden Beispiel. "
    "Die Schwächen von CH sind feste Kosten (nach einer Verkehrsänderung muss neu vorgerechnet werden) und die Abhängigkeit von der Form des Netzes. Das Verfahren folgt dem Kapitel 4.3.4 in *Optimization Algorithms* (A. Khamis) und der Arbeit von Geisberger et al. (2008); "
    "Netze und Zahlen dieser Demo sind eigene Graphen und Messungen sowie OpenStreetMap-Daten."
)

with st.expander("So funktioniert Contraction Hierarchies", expanded=True):
    st.markdown(
        """
1. **Ordnen:** jeder Knoten bekommt eine Wichtigkeit, zum Beispiel die **Kantendifferenz**: wie viele Abkürzungen würde sein Zusammenziehen kosten, verglichen mit den Kanten, die dabei wegfallen? Wer wenig kostet, kommt zuerst dran.
2. **Zusammenziehen:** für jedes Paar Eingangs- und Ausgangsnachbar $u \\to v \\to w$ des Knotens $v$ läuft eine kleine **Zeugensuche** (Dijkstra ohne $v$). Findet sie einen Weg $u \\to w$ höchstens so teuer wie über $v$, ist nichts zu tun - sonst kommt die **Abkürzung** $u \\to w$ hinzu (Kosten der beiden Kanten, Mittelknoten $v$).
   $v$ bekommt seinen Rang, alle Kanten von und zu $v$ gehören ab jetzt zur Hierarchie.
3. **Abfrage:** vorwärts von $s$ nur auf Kanten zu **höherem** Rang, rückwärts von $t$ ebenfalls nur aufwärts. Die beste Summe an einem Knoten, den beide erreichen, ist die Route - sie hat eine Form wie ein Berg: erst hinauf, dann hinunter.
4. **Auspacken:** jede Abkürzung der Route wird durch ihren Mittelknoten ersetzt, bis nur Originalkanten übrig sind.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

# Zahl der Knoten des gewählten Netzes (für die Ordnung "zufällig", die bei großen Netzen Minuten braucht)
_net_now = st.session_state["net_select"]
_n_now = {"small": 8, "toronto": 10153}.get(_net_now, int(st.session_state.get("side_slider", st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))) ** 2 if _net_now == "city" else int(st.session_state.get("nodes_slider", st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))))
order_options = [o for o in C.ORDER_LABELS if o != "random" or _n_now <= C.RANDOM_ORDER_MAX_NODES]
if st.session_state["order_select"] not in order_options:
    st.session_state["order_select"] = C.DEFAULT_ORDER

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Klein und fest (Schritt für Schritt), erzeugt (Stadtnetz, Zufallsnetz) oder echtes Autonetz der Toronto-Innenstadt (OpenStreetMap, 10 153 Kreuzungen, 26 985 gerichtete Kanten). Alle Kosten sind ganze Zahlen. "
             "Die Vorberechnung läuft beim ersten Aufruf eines Netzes live (Sekunden) und wird dann gemerkt.",
    )
    if net_key == "city":
        seed_widget("side_slider")
        side = st.slider("Kreuzungen je Seite", *bounds("side_slider"), key="side_slider",
                         help="Größe des Stadtnetzes. Gewinn der CH-Abfrage gegen die bidirektionale Suche (Median über Paare, Mittel über fünf Netze) bei 10 / 20 / 30 Kreuzungen je Seite: 0.8-fach / 1.1-fach / 1.4-fach - größere Netze haben mehr Hierarchie.")
        st.session_state[KEPT["side_slider"]] = side
        seed_widget("reach_slider")
        reach = st.slider("Reichweite der Straßen [Blocklängen]", *bounds("reach_slider"), key="reach_slider", step=0.1,
                          help="Wie weit eine Straße zwischen zwei Kreuzungen reichen darf (1 = nur Nachbarn im Raster). Gewinn gegen die bidirektionale Suche bei 1.0 / 1.5 / 2.3 / 3.2: 3.3-fach / 1.9-fach / 1.1-fach / 0.9-fach - dichte Netze verlieren ihre Hierarchie.")
        st.session_state[KEPT["reach_slider"]] = reach
        seed_widget("spread_slider")
        spread = st.slider("Streuung der Kosten", *bounds("spread_slider"), key="spread_slider", step=0.25,
                           help="Kosten einer Straße = Länge × (1 + Streuung × Zufall), gerundet auf ganze Meter. Gewinn gegen die bidirektionale Suche bei 0 / 1 / 3: 1.0-fach / 1.1-fach / 1.3-fach - ungleiche Kosten helfen der Hierarchie etwas.")
        st.session_state[KEPT["spread_slider"]] = spread
        seed_widget("blocked_slider")
        blocked = st.slider("Gesperrte Straßen [%]", *bounds("blocked_slider"), key="blocked_slider",
                            help="Anteil der gesperrten Straßen (das Netz bleibt zusammenhängend). Gewinn gegen die bidirektionale Suche bei 0 / 20 / 60 %: 1.1-fach / 1.1-fach / 1.5-fach - im ausgedünnten Netz trägt die Hierarchie etwas mehr.")
        st.session_state[KEPT["blocked_slider"]] = blocked
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
        reach = float(st.session_state.get(KEPT["reach_slider"], C.DEFAULT_REACH))
        spread = float(st.session_state.get(KEPT["spread_slider"], C.DEFAULT_SPREAD))
        blocked = int(st.session_state.get(KEPT["blocked_slider"], C.DEFAULT_BLOCKED))
    if net_key == "random":
        seed_widget("nodes_slider")
        nodes = st.slider("Knoten", *bounds("nodes_slider"), key="nodes_slider", step=100, help="Anzahl der Knoten des Zufallsnetzes.")
        st.session_state[KEPT["nodes_slider"]] = nodes
        seed_widget("degree_slider")
        degree = st.slider("Mittlerer Grad", *bounds("degree_slider"), key="degree_slider", step=0.5, help="Wie viele Nachbarn ein Knoten im Mittel hat. Gewinn der CH-Abfrage gegen die bidirektionale Suche (Median über Paare, Mittel über fünf Netze, 600 Knoten) bei Grad 2.5 / 4 / 6: 1.2-fach / 0.6-fach / 0.4-fach - je dichter das Netz, desto weniger Hierarchie.")
        st.session_state[KEPT["degree_slider"]] = degree
    else:
        nodes = int(st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))
        degree = float(st.session_state.get(KEPT["degree_slider"], C.DEFAULT_DEGREE))
    if net_key != "small":
        seed_widget("distance_slider")
        distance = st.slider("Entfernung Start–Ziel [%]", *bounds("distance_slider"), key="distance_slider",
                             help="Welcher Knoten das Ziel ist: der, dessen Entfernung vom Start in der Rangfolge aller erreichbaren Knoten bei diesem Prozentwert liegt (100 = der am weitesten entfernte).")
        st.session_state[KEPT["distance_slider"]] = distance
    else:
        distance = int(st.session_state.get(KEPT["distance_slider"], C.DEFAULT_DISTANCE))
    order = st.selectbox("Ordnung der Knoten", order_options, key="order_select", format_func=lambda k: C.ORDER_LABELS[k],
                         help="Nach welcher Wichtigkeit die Knoten zusammengezogen werden. Die Richtigkeit hängt nie davon ab, nur die Zahl der Abkürzungen und der Aufwand. \"zufällig\" gibt es nur für kleine Netze (bis 500 Knoten): bei großen "
                              "explodieren die Abkürzungen und die Vorberechnung läuft Minuten. Auf dem Stadtnetz (16 × 16) braucht die zufällige Ordnung 2.0 Abkürzungen je Kante, die Kantendifferenz 0.6-0.7.")
    witness = st.selectbox("Zeugensuche (Knoten je Suche)", list(C.WITNESS_OPTIONS), key="witness_select", format_func=lambda k: C.WITNESS_LABELS[k],
                           help="Wie viele Knoten eine Zeugensuche höchstens festlegt. Weniger Zeugen kosten nur Abkürzungen, nie Richtigkeit: in Toronto bei 1 / 5 / 20 / 200 Knoten 3.1 / 2.2 / 1.5 / 1.4 Abkürzungen je Kante.")
    if net_key in ("city", "random"):
        seed_widget("seed_input")
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - es gibt nichts zu erzeugen.")

sync_query_params({"net_select": net_key, "side_slider": int(side), "reach_slider": round(float(reach), 1), "spread_slider": round(float(spread), 2), "blocked_slider": int(blocked),
                   "nodes_slider": int(nodes), "degree_slider": round(float(degree), 1), "distance_slider": int(distance), "order_select": order, "witness_select": int(witness), "seed_input": int(seed)})

# nicht zum Netz gehörende Regler ändern das Netz nicht: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
if net_key in C.FIXED_NETS:
    params = (net_key, C.DEFAULT_SIDE, C.DEFAULT_REACH, C.DEFAULT_SPREAD, C.DEFAULT_BLOCKED, C.DEFAULT_NODES, C.DEFAULT_DEGREE, C.DEFAULT_SEED)
elif net_key == "city":
    params = (net_key, int(side), round(float(reach), 1), round(float(spread), 2), int(blocked), C.DEFAULT_NODES, C.DEFAULT_DEGREE, int(seed))
else:
    params = (net_key, C.DEFAULT_SIDE, C.DEFAULT_REACH, C.DEFAULT_SPREAD, C.DEFAULT_BLOCKED, int(nodes), round(float(degree), 1), int(seed))
distance_used = C.DEFAULT_DISTANCE if net_key == "small" else int(distance)
with st.spinner("Rechne vor (Zusammenziehen aller Knoten) ..." if net_key != "small" else "Rechne..."):
    h = _hierarchy(params, order, int(witness))
    a = _analysis(params, order, int(witness), distance_used)
net, m = a.net, a.metrics
g = net.graph
small = bool(g.names)
view_key = (params, order, int(witness), distance_used)

# --- Vorberechnung: Zusammenziehen --------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Vorberechnung: Knoten zusammenziehen")
if small:
    if st.session_state.get("ch_contract_owner") != view_key:
        st.session_state["ch_contract"] = g.n
        st.session_state["ch_contract_owner"] = view_key
    step_c = st.slider("Zusammengezogene Knoten", 0, g.n, key="ch_contract",
                       help="Wie viele Knoten schon zusammengezogen sind: 0 = das Netz vorher, ganz rechts = alle Knoten haben ihren Rang. Orange gestrichelt: eingefügte Abkürzungen mit ihren Kosten.")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_contraction(net, h, step_c), width="stretch", key="contraction_chart")
    with c2:
        if step_c == 0:
            st.markdown("**Noch nichts zusammengezogen.** Die Ordnung steht: als Erster kommt " + g.names[h.order[0]] + " dran (geringste Wichtigkeit).")
        else:
            v, r, entries = h.log[step_c - 1]
            st.markdown(f"**Schritt {step_c}: {g.names[v]} bekommt Rang {r + 1}.**")
            seen_pairs, lines = set(), []
            for u, w, via, wit, inserted in entries:
                key = (min(u, w), max(u, w))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                lines.append(f"- {g.names[u]} ↔ {g.names[w]} über {g.names[v]}: {via:g} " + (f"→ **Abkürzung eingefügt** ({via:g})" if inserted else f"→ Zeuge mit {wit:g} - **keine Abkürzung nötig**"))
            st.markdown("\n".join(lines) if lines else "Keine zwei Nachbarn im Restnetz - nichts zu tun.")
        st.markdown(f"**Bisher {sum(1 for _, _, en in h.log[:step_c] for e in en if e[4]) // 2 if not g.directed else sum(1 for _, _, en in h.log[:step_c] for e in en if e[4])} Abkürzungen** (je Paar einmal).")
else:
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Abkürzungen", _num(m["n_shortcuts"]), delta=f"{m['shortcut_ratio']:.2f} je Originalkante", delta_color="off", help=f"Bei {_num(m['n_arcs'])} gerichteten Kanten im Netz.")
    b2.metric("Zeugensuchen", _num(h.counters["witness_searches"]), delta=f"{_num(h.counters['witness_settled'])} Knoten festgelegt", delta_color="off", help="Kleine Dijkstra-Suchen, die entscheiden, ob eine Abkürzung nötig ist.")
    b3.metric("Vorberechnung", f"{h.seconds:.1f} s", help="Messwert dieses Rechners für das einmalige Vorrechnen (reines Python).")
    b4.metric("Knoten", _num(g.n), delta=f"{len(h.added) and int(sum(1 for x in h.added if x > 0))} mit Abkürzung", delta_color="off", help="Knoten, bei deren Zusammenziehen mindestens eine Abkürzung eingefügt werden musste.")
    st.plotly_chart(build_added(h), width="stretch", key="added_chart")
    _dec = np.array_split(np.array(h.added, dtype=float), 10)
    _means = [float(d.mean()) for d in _dec]
    st.caption(f"Mittlere Zahl eingefügter Abkürzungen je zusammengezogenem Knoten: {_means[0]:.1f} im ersten Zehntel der Reihenfolge, {_means[-1]:.1f} im letzten, am höchsten {max(_means):.1f} (im {int(np.argmax(_means)) + 1}. Zehntel). "
               "Die wichtigen Knoten am Ende bilden den Kern der Hierarchie; wie viel Aufwand dort entsteht, hängt vom Netz ab.")

st.markdown("---")

# --- Die Abfrage in Aktion ------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Abfrage in Aktion")
last_step = len(a.ch.steps)
if st.session_state.get("ch_step_owner") != view_key:
    st.session_state["ch_step"] = last_step
    st.session_state["ch_step_owner"] = view_key
step_col, play_col = st.columns([5, 2])
with step_col:
    if last_step > 1:
        step = st.slider("Festlegungen der CH-Abfrage (abwechselnd vorwärts und rückwärts)", 0, last_step, key="ch_step",
                         help="Wie viele Knoten die CH-Abfrage schon festgelegt hat: 0 = nur Start und Ziel, ganz rechts = fertig, die Route erscheint.")
    else:
        step = last_step
        st.caption("Start und Ziel sind derselbe Knoten - es gibt keine Festlegung.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
o1, o2 = st.columns(2)
show_uni = o1.checkbox("Fläche von Dijkstra einblenden (grau)", value=True, key="show_uni", help="Grau: alle Knoten, die einseitiges Dijkstra bis zum Ziel festgelegt hätte.")
show_bi = o2.checkbox("Fläche der bidirektionalen Suche einblenden (hell)", value=True, key="show_bi", help="Hellblau und hellgrün: die beiden Flächen der bidirektionalen Suche aus dem vorigen Stück.")
view_slot = st.empty()


def _render(current):
    with view_slot.container():
        st.plotly_chart(build_query(net, a, current, show_uni, show_bi, height=460 if small else 540), width="stretch", key=f"query_chart_{current}")


if auto_play:
    n_frames = min(max(last_step, 1), 40)
    for k in sorted({int(round(x)) for x in np.linspace(0, last_step, n_frames + 1)}):
        _render(k)
        time.sleep(min(0.6, 6.0 / n_frames))
    step = last_step
else:
    _render(step)
st.caption(net.note + (" Karte: © [OpenStreetMap-Mitwirkende](https://www.openstreetmap.org/copyright), Daten unter der [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/1-0/)." if net.key == "toronto" else ""))

st.markdown("---")

# --- Erst vorrechnen, dann blitzschnell fragen ----------------------------------------------------------------------------------------------

st.markdown("## 🎯 Erst vorrechnen, dann blitzschnell fragen")
st.caption("**Festgelegt** = Knoten, die eine Suche bis zum Ziel festlegt - der Aufwand einer Suche, plattformfest. Laufzeiten stehen nur als Messwerte im Vergleich unten. Die drei Verfahren liefern dieselben Kosten (in jedem Lauf geprüft).")
if not m["reachable"]:
    st.warning("⚠️ Das Ziel ist vom Start aus nicht erreichbar - alle drei Verfahren melden das ausdrücklich, statt eine Route zu erfinden.")
else:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CH-Abfrage", _num(m["settled_ch"]), delta=f"vorwärts {_num(m['settled_ch_f'])} · rückwärts {_num(m['settled_ch_b'])}", delta_color="off", help="Festgelegte Knoten beider Seiten der CH-Abfrage.")
    m2.metric("Bidirektional", _num(m["settled_bi"]), delta=f"{m['gain_bi']:.1f}× so viele wie CH", delta_color="off", help="Die bidirektionale Suche des vorigen Stücks.")
    m3.metric("Dijkstra", _num(m["settled_uni"]), delta=f"{m['gain_uni']:.1f}× so viele wie CH", delta_color="off")
    m4.metric("Kosten", _cost(net, m["cost_ch"]), delta=f"{m['hops']} Kanten, in der Hierarchie {m['packed_hops']} Sprünge", delta_color="off",
              help=f"Die Route hat {m['hops']} Originalkanten; in der Hierarchie sind es {m['packed_hops']} Kanten, davon {m['shortcuts_used']} Abkürzungen. Treffpunkt: Rang {m['peak_rank'] + 1} von {_num(m['n'])}.")
    code = ev.verdict(a)
    be = f" Die Vorberechnung ({h.seconds:.1f} s) hat sich gegenüber Dijkstra nach etwa {m['break_even']:,.0f} Anfragen ausgeglichen (Messwert dieses Laufs).".replace(",", ".") if np.isfinite(m["break_even"]) and m["break_even"] > 0 else ""
    if code == "wrong":
        st.warning("⚠️ Die Kosten der drei Verfahren stimmen nicht überein - das dürfte nicht passieren.")
    elif code == "faster":
        st.success(f"✅ Dieselben Kosten ({_cost(net, m['cost_ch'])}), aber die CH-Abfrage legt nur {_num(m['settled_ch'])} Knoten fest - **{m['gain_bi']:.1f}-fach weniger** als die bidirektionale Suche ({_num(m['settled_bi'])}) und {m['gain_uni']:.1f}-fach weniger als Dijkstra ({_num(m['settled_uni'])}).{be}")
    else:
        st.info(f"ℹ️ Hier lohnt sich die Hierarchie nicht: die CH-Abfrage legt {_num(m['settled_ch'])} Knoten fest, die bidirektionale Suche {_num(m['settled_bi'])} (Dijkstra {_num(m['settled_uni'])}). "
                + ("Bei acht Orten gibt es nichts abzukürzen - der Nutzen der Vorberechnung zeigt sich erst in großen Netzen mit Hierarchie (Toronto)." if small else
                   f"Die {_num(m['n_shortcuts'])} Abkürzungen kosten Platz und Vorrechenzeit, ohne der Abfrage in diesem Netz zu helfen (siehe den Vergleich der Netztypen unten)."))

    st.markdown("**Nicht nur dieses eine Paar**")
    ps = _pair_stats(params, order, int(witness), C.PAIRS)
    if ps["n_pairs"]:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Gewinn gegen bidirektional (Median)", f"{ps['gain_bi_median']:.1f}×", delta=f"10 %–90 %: {ps['gain_bi_p10']:.1f}×–{ps['gain_bi_p90']:.1f}×", delta_color="off",
                  help=f"Festgelegte Knoten der bidirektionalen Suche geteilt durch die der CH-Abfrage, über {ps['n_pairs']} zufällige erreichbare Paare.")
        p2.metric("Gewinn gegen Dijkstra (Median)", f"{ps['gain_uni_median']:.1f}×")
        p3.metric("Paare ohne Gewinn", _pct(ps["share_no_gain_bi"]), help="Anteil der Paare, in denen die CH-Abfrage nicht weniger festlegt als die bidirektionale Suche.")
        p4.metric("Falsche Kosten", f"{ps['wrong']}", help="Zahl der Paare, in denen CH andere Kosten als Dijkstra liefert - immer 0.")
        st.plotly_chart(build_gain_hist(ps["gain_bi"]), width="stretch", key="gain_hist")
        st.caption(f"{ps['n_pairs']} zufällige Start-Ziel-Paare im gewählten Netz; Mediane der festgelegten Knoten: Dijkstra {_num(int(ps['median_uni']))}, bidirektional {_num(int(ps['median_bi']))}, CH {_num(int(ps['median_ch']))}.")

st.markdown("---")

# --- Vergleich -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Dijkstra, bidirektional und CH im Vergleich"):
    if m["reachable"]:
        st.table({"Verfahren": ["Dijkstra", "bidirektional", "CH-Abfrage"], "Festgelegte Knoten": [_num(m["settled_uni"]), _num(m["settled_bi"]), _num(m["settled_ch"])],
                  "Laufzeit [ms]": [f"{a.seconds['uni'] * 1000:.2f}", f"{a.seconds['bi'] * 1000:.2f}", f"{a.seconds['ch'] * 1000:.2f}"]})
        st.table({"Vorberechnung": ["Abkürzungen", "Zeugensuchen", "Wichtigkeitsberechnungen"], "Anzahl": [_num(h.counters["shortcuts"]), _num(h.counters["witness_searches"]), _num(h.counters["priority_evaluations"])],
                  "Knoten in Suchen": ["–", _num(h.counters["witness_settled"]), _num(h.counters["sim_settled"])]})
    st.caption("Die Laufzeiten sind Messwerte dieses Laufs (reines Python, ein Lauf) und schwanken. Die CH-Abfrage braucht dafür die Vorberechnung und ihre Hierarchie im Speicher: neben den Originalkanten die Abkürzungen (Anzahl oben).")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wo eine Hierarchie da ist – und wo nicht")
if st.button("Stadtnetz, Toronto und Zufallsnetz vergleichen (dauert etwa 20 Sekunden)", key="netcmp_start"):
    st.session_state["netcmp_on"] = True
if st.session_state.get("netcmp_on"):
    with st.spinner("Rechne drei Netztypen vor (Stadtnetz und Zufallsnetz × 5 Datensätze, Toronto einmal) ..."):
        ncr = _net_comparison()
    by = {r["net"]: r for r in ncr}
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_net_comparison(ncr), width="stretch", key="netcmp_chart")
    c2.table({"Netz": [{"city": "Stadtnetz", "toronto": "Toronto", "random": "Zufallsnetz"}[r["net"]] for r in ncr], "Abkürzungen je Kante": [f"{r['shortcut_ratio']:.2f}" for r in ncr],
              "Gewinn gegen bidirektional": [f"{r['gain_bi']:.1f}×" for r in ncr]})
    st.caption(f"Median der festgelegten Knoten über 30 zufällige Paare; Stadtnetz und Zufallsnetz gemittelt über 5 feste Datensätze, Toronto einmal. Im echten **Straßennetz** ({by['toronto']['gain_bi']:.0f}× gegen die bidirektionale Suche) trägt die Hierarchie - "
               f"üblicherweise erklärt durch wenige Hauptstraßen und viele Nebenstraßen. Im **Stadtnetz-Gitter** ({by['city']['gain_bi']:.1f}×) fehlt sie fast, im **Zufallsnetz** ({by['random']['gain_bi']:.1f}×) ganz: dort legt die CH-Abfrage im Median sogar mehr Knoten fest als die bidirektionale Suche, "
               f"und die Vorberechnung fügt {by['random']['shortcut_ratio']:.1f} Abkürzungen je Kante ein.")

st.markdown("---")

st.subheader("🔬 Die Ordnung der Knoten")
if st.button("Vier Ordnungen auf dem Stadtnetz vergleichen (dauert etwa 20 Sekunden)", key="order_start"):
    st.session_state["order_on"] = True
if st.session_state.get("order_on"):
    with st.spinner("Rechne vier Ordnungen × 5 Datensätze vor ..."):
        orows = _order_comparison()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_order(orows), width="stretch", key="order_chart")
    names = {"edge_difference": "Kantendifferenz (Nachbarn)", "lazy_edge_difference": "Kantendifferenz (träge)", "degree": "nur Grad", "random": "zufällig"}
    c2.table({"Ordnung": [names[r["order"]] for r in orows], "Abkürzungen je Kante": [f"{r['shortcut_ratio']:.2f}" for r in orows], "CH festgelegt (Median)": [f"{r['median_ch']:.0f}" for r in orows]})
    ob = {r["order"]: r for r in orows}
    st.caption(f"Stadtnetz mit 16 × 16 Kreuzungen, Mittel über 5 Datensätze. Alle vier Ordnungen liefern exakte Kosten. Die **zufällige** Reihenfolge braucht {ob['random']['shortcut_ratio'] / ob['lazy_edge_difference']['shortcut_ratio']:.1f}-mal so viele Abkürzungen wie die träge Kantendifferenz "
               f"({ob['random']['shortcut_ratio']:.1f} gegen {ob['lazy_edge_difference']['shortcut_ratio']:.1f} je Kante); in diesem kleinen Netz merkt die Abfrage davon wenig ({ob['random']['median_ch']:.0f} gegen {ob['lazy_edge_difference']['median_ch']:.0f} festgelegte Knoten). "
               "In einem großen Netz ist die zufällige Ordnung nicht mehr zu gebrauchen: die Vorberechnung läuft Minuten, deshalb gibt es sie dort nicht zur Auswahl.")

st.markdown("---")

st.subheader("🔬 Zeugensuche begrenzen")
if st.button("Vier Zeugengrenzen auf Toronto vergleichen (dauert etwa 10 Sekunden)", key="witness_start"):
    st.session_state["witness_on"] = True
if st.session_state.get("witness_on"):
    with st.spinner("Rechne Toronto viermal vor ..."):
        wrows = _witness_comparison()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_witness(wrows), width="stretch", key="witness_chart")
    c2.table({"Zeugensuche bis": [str(r["witness"]) for r in wrows], "Abkürzungen je Kante": [f"{r['shortcut_ratio']:.2f}" for r in wrows], "Falsche Kosten": [str(r["wrong"]) for r in wrows]})
    st.caption(f"Toronto Innenstadt, 30 zufällige Paare. Eine Zeugensuche darf abbrechen, bevor sie einen Zeugen findet - dann kommt eine Abkürzung dazu, die vielleicht überflüssig ist. Das kostet Platz (Grenze 1: {wrows[0]['shortcut_ratio']:.1f} Abkürzungen je Kante gegen {wrows[-1]['shortcut_ratio']:.1f} bei 200), "
               "aber nie Richtigkeit: alle Abfragen bleiben exakt.")

st.markdown("---")

st.subheader("🔬 Der Verkehr ändert sich")
if st.button("Kosten ändern und die alte Hierarchie fragen (dauert etwa 10 Sekunden)", key="stale_start"):
    st.session_state["stale_on"] = True
if st.session_state.get("stale_on"):
    with st.spinner("Ändere Kosten und frage die alte Hierarchie ..."):
        sw = _stale_weights()
    st.plotly_chart(build_stale(sw["rows"]), width="stretch", key="stale_chart")
    r0 = sw["rows"][0]
    st.caption(f"Toronto Innenstadt, 40 zufällige Paare, verdoppelte Kosten bei einem Anteil der Kanten. Die Hierarchie kennt nur die alten Kosten: schon bei {r0['fraction']:.0%} teureren Kanten sind bei {r0['share_wrong_cost']:.0%} der Paare die berichteten Kosten falsch und bei {r0['share_suboptimal']:.0%} "
               f"die Route nicht mehr die kürzeste (im Mittel {r0['excess_mean']:.1%} zu lang). Neu vorrechnen dauert hier {sw['rebuild']['seconds']:.1f} Sekunden (Messwert) und macht alle {sw['rebuild']['correct']} von {sw['rebuild']['pairs']} Paaren wieder exakt.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Kosten ändern sich nicht** | Toronto: schon bei 2 % teureren Kanten stimmen bei 78 % der Paare die berichteten Kosten nicht mehr, bei 60 % ist die Route nicht mehr die kürzeste; bei 20 % teureren Kanten sind es 100 % und 95 %. Neu vorrechnen dauert Sekunden. | angepasste Hierarchien für wechselnde Kosten (nicht gebaut) |
| **Das Netz hat eine Hierarchie** | Toronto: im Median 21-fach weniger festgelegte Knoten als bidirektional. Stadtnetz-Gitter: 1.1-fach. Zufallsnetz: 0.5-fach - die CH-Abfrage legt fast doppelt so viele Knoten fest, die Vorberechnung fügt 1.9 Abkürzungen je Kante ein. | (die Form des Netzes entscheidet) |
| **Vorrechnen lohnt sich** | Die Vorberechnung kostet Sekunden und Speicher für die Abkürzungen (Toronto: 1.4 je Kante); erst nach vielen Anfragen ist sie ausgeglichen (Messwert im Kernabschnitt); für wenige Anfragen lohnt sie sich nicht. | |
| **Nur Start-Ziel-Anfragen** | CH beantwortet ein Paar nach dem anderen; für die Entfernungen zwischen **allen** Paaren gibt es andere Verfahren. | **Floyd-Warshall**, **Johnson** |
"""
)
st.caption("Die Nachbarn der Kürzeste-Wege-Linie (noch nicht gebaut): Bellman-Ford, Floyd-Warshall, Johnson und Mehrkriterien-Routing. Bereits gebaut: Breitensuche, Dijkstra und bidirektionale Suche.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Gerichteter Graph $G=(V,E)$ mit Kosten $c_e \ge 0$. Eine **Ordnung** (Rang) $r: V \to \{1,\dots,n\}$ bestimmt die Reihenfolge des Zusammenziehens.

**Zusammenziehen.** Beim Zusammenziehen von $v$ (im Restgraphen $G_i$) wird für jedes Paar $u \to v \to w$ mit $u \ne w$ geprüft, ob es in $G_i \setminus \{v\}$ einen Weg $u \leadsto w$ mit Kosten $\le c(u,v)+c(v,w)$ gibt (**Zeuge**).
Wenn nicht, kommt die Kante $(u,w)$ mit Kosten $c(u,v)+c(v,w)$ und Mittelknoten $v$ hinzu. $G_{i+1}$ ist $G_i \setminus \{v\}$ plus diese Abkürzungen. **Invariante:** die Entfernungen zwischen den verbleibenden Knoten sind in $G_{i+1}$ dieselben wie in $G_i$.
Eine Zeugensuche darf nach beliebig wenigen Schritten abbrechen: dann fehlt ein Zeuge, und eine überflüssige Abkürzung entsteht - die Invariante bleibt.

**Hierarchie.** $G^{+}$ = alle Kanten von $G$ und alle Abkürzungen; jede Kante liegt beim Zusammenziehen ihres niedriger gerankten Endknotens fest. $G^{\uparrow}$ enthält die Kanten $(u,v)$ mit $r(u)<r(v)$, $G^{\downarrow}$ die mit $r(u)>r(v)$.

**Abfrage und Richtigkeit.** Die Abfrage läuft vorwärts auf $G^{\uparrow}$ ab $s$ und rückwärts auf den umgedrehten Kanten von $G^{\downarrow}$ ab $t$. **Beweisidee:** zu einer kürzesten Route $P$ von $s$ nach $t$ sei $x$ der Knoten mit dem höchsten Rang auf $P$.
Liegt auf $P$ ein Knoten $y$ mit niedrigerem Rang als beide Nachbarn $u,w$ auf $P$, wurde $y$ vor $u$ und $w$ zusammengezogen: entweder gibt es die Abkürzung $u \to w$ mit gleichen Kosten oder einen Zeugen, der $P$ nicht verlängert. Ersetzt man wiederholt so, entsteht eine Route derselben Länge,
die erst steigt und dann fällt, mit $x$ als Spitze. Diese Route findet die Abfrage: der Teil $s \leadsto x$ liegt in $G^{\uparrow}$, der Teil $x \leadsto t$ (rückwärts gelesen) ebenfalls. Der Abbruch, wenn der kleinste Schlüssel einer Seite die beste bisherige Route erreicht, ist wie bei der bidirektionalen Suche zulässig.
Kein anderer Knoten kann eine kürzere Route liefern, weil jede Route über $G^{+}$ eine echte Route in $G$ ist (Auspacken der Abkürzungen).

**Wichtigkeit.** Kantendifferenz $ED(v)=\#\text{Abkürzungen}(v)-\bigl(\deg_{\text{ein}}(v)+\deg_{\text{aus}}(v)\bigr)$, dazu die Zahl schon zusammengezogener Nachbarn. Die träge Variante berechnet die Wichtigkeit erst beim Entnehmen neu.
**Warum Gitter und Zufallsgraphen kaum eine Hierarchie haben** (übliche Erklärung aus der Literatur, hier nur an den Messungen oben belegt, nicht getrennt geprüft): in einem Straßennetz mit wenigen Hauptstraßen liegen viele Wege über wenige Knoten (kleine Trennmengen, kleine "Autobahndimension"); in Gittern und Zufallsgraphen gibt es solche Engstellen nicht,
jedes Zusammenziehen erzeugt viele Abkürzungen, und die Abfrage muss viele Knoten aufwärts absuchen.

Implementiert in `ch_graph.py` (CSR-Graph), `ch_queues.py` (Warteschlangen), `ch_bd.py` (Dijkstra und bidirektionale Suche aus dem vorigen Stück), `ch_algorithm.py` (Zusammenziehen, Zeugensuche, Ordnungen, Abfrage, Auspacken), `ch_scenario.py` (Netze), `ch_evaluation.py` (Kennzahlen, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
