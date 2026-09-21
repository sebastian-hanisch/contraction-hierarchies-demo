"""Konstanten, Grenzen der Regler und Presets. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 100.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.25                      # Lageabweichung der Kreuzungen in Blocklängen

NETS = ("small", "city", "toronto", "random")
NET_LABELS = {
    "small": "🔀 Kleines Netz (Schritt für Schritt)",
    "city": "🏙️ Stadtnetz (erzeugt)",
    "toronto": "🍁 Toronto Innenstadt (OpenStreetMap)",
    "random": "🕸️ Zufallsnetz (erzeugt)",
}
FIXED_NETS = ("small", "toronto")   # keine Größenregler; das kleine Netz hat außerdem eine feste Aufgabe

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 6, 30, 20
REACH_MIN, REACH_MAX, DEFAULT_REACH = 1.0, 3.2, 2.3
SPREAD_MIN, SPREAD_MAX, DEFAULT_SPREAD = 0.0, 3.0, 1.0
BLOCKED_MIN, BLOCKED_MAX, DEFAULT_BLOCKED = 0, 60, 20          # Prozent der Straßen
NODES_MIN, NODES_MAX, DEFAULT_NODES = 200, 1000, 600
DEGREE_MIN, DEGREE_MAX, DEFAULT_DEGREE = 2.5, 6.0, 4.0
DISTANCE_MIN, DISTANCE_MAX, DEFAULT_DISTANCE = 10, 100, 60     # Prozent: Rang der Entfernung des Ziels vom Start
DEFAULT_SEED = 7
DEFAULT_NET = "small"

# Ordnung der Knoten beim Zusammenziehen; "random" nur für kleine Netze (bei großen explodieren die Abkürzungen, die Vorberechnung läuft Minuten)
ORDER_LABELS = {"lazy_edge_difference": "Kantendifferenz, träge neu geprüft", "edge_difference": "Kantendifferenz, Nachbarn neu bewertet", "degree": "nur Grad", "random": "zufällig"}
DEFAULT_ORDER = "lazy_edge_difference"
RANDOM_ORDER_MAX_NODES = 500
WITNESS_OPTIONS = (1, 5, 20, 200)                              # festgelegte Knoten je Zeugensuche
WITNESS_LABELS = {1: "1 (fast keine Zeugen)", 5: "5", 20: "20", 200: "200 (Standard)"}
DEFAULT_WITNESS = 200
SIM_LIMIT_FOR = {1: 1, 5: 5, 20: 10, 200: 50}                  # Probelauf-Grenze beim Abschätzen der Wichtigkeit

SWEEP_SEEDS = tuple(range(100000, 100005))
PAIRS = 100                        # zufällige Start-Ziel-Paare je Netz für die Verteilungen

COLORS = {"uni": "#9e9e9e", "bi": "#1f77b4", "bi2": "#2ca02c", "ch": "#d62728", "route": "#111111", "start": "#111111", "goal": "#ff7f0e", "peak": "#9467bd", "shortcut": "#ff7f0e"}

# Jedes Preset setzt alle Regler; nicht zum Netz gehörende Regler sind dort ausgeblendet und werden auf die Standardwerte gesetzt.
_BASE = dict(side=DEFAULT_SIDE, reach=DEFAULT_REACH, spread=DEFAULT_SPREAD, blocked=DEFAULT_BLOCKED, nodes=DEFAULT_NODES, degree=DEFAULT_DEGREE, distance=DEFAULT_DISTANCE,
             order=DEFAULT_ORDER, witness=DEFAULT_WITNESS, seed=DEFAULT_SEED)
PRESETS = {
    "🔀 Kleines Netz": {**_BASE, "net": "small"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city"},
    "🍁 Toronto Innenstadt": {**_BASE, "net": "toronto"},
    "🕸️ Zufallsnetz": {**_BASE, "net": "random"},
}
PRESET_HELP = {
    "🔀 Kleines Netz": "Acht Orte mit zwei Ortsgruppen und einer schnellen Verbindung: beim Zusammenziehen entstehen 6 gerichtete Abkürzungen (drei Paare), an mehreren Stellen genügt ein Umweg als Zeuge. Bei acht Knoten lohnt sich die Vorberechnung noch nicht - hier geht es ums Verstehen.",
    "🏙️ Stadtnetz": "Erzeugtes Stadtnetz (20 × 20 Kreuzungen, Reichweite 2.3): für das gezeigte Paar legt die CH-Abfrage 51 Knoten fest, die bidirektionale Suche 126 und Dijkstra 241 - im Mittel über fünf Netze aber nur etwa 1.1-fach besser als bidirektional: ein Gitter hat kaum Hierarchie.",
    "🍁 Toronto Innenstadt": "Echtes Autonetz (10 153 Kreuzungen, 26 985 Kanten): für das gezeigte Paar legt die CH-Abfrage nur 176 Knoten fest, die bidirektionale Suche 3 547 und Dijkstra 6 093. Die Vorberechnung dauert etwa 3 Sekunden und fügt 37 340 Abkürzungen ein (1.4 je Kante).",
    "🕸️ Zufallsnetz": "Erzeugter Zufallsgraph (600 Knoten, im Mittel 4 Nachbarn) ohne Hierarchie: die Vorberechnung fügt 4 293 Abkürzungen ein (1.8 je Kante), und die CH-Abfrage legt für das gezeigte Paar mit 51 Knoten mehr fest als die bidirektionale Suche (41).",
}
