# Contraction Hierarchies – erst vorrechnen, dann blitzschnell fragen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-contraction-hierarchies-demo.streamlit.app/)**

Viertes Stück der **Kürzeste-Wege-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Fortsetzung der [Demo zur bidirektionalen Suche](../bidirectional-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – **Contraction Hierarchies (CH)** – an einem wachsenden Beispiel.
Die bidirektionale Suche beginnt jede Anfrage von vorn und legt im Toronto-Netz für ein Paar mehrere tausend Knoten fest. CH **rechnet einmal vor**: die Knoten werden nach Wichtigkeit geordnet und von unten nach oben **zusammengezogen**,
wobei **Abkürzungen** die Wege über den entfernten Knoten ersetzen – außer ein **Zeuge** (ein gleich kurzer anderer Weg) macht sie überflüssig. Die Abfrage sucht dann von beiden Enden nur noch **aufwärts** in der Hierarchie.
Der Preis: Vorrechnen, feste Kosten – und ein Netz, das überhaupt eine Hierarchie hat.

**Einordnung in die Reihe (die Kanten des Graphen):** CH setzt an der Schwäche der bidirektionalen Suche an ("jede Anfrage beginnt von vorn"), seine Abfrage ist selbst eine bidirektionale Dijkstra-Suche.
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)                              [gebaut]
  └─ dijkstra-demo (Kosten korrekt, blind in alle Richtungen)               [gebaut]
       ├─ bidirectional-demo (von beiden Enden) → Contraction Hierarchies   [gebaut]
       │                                            └─ contraction-hierarchies-demo   [dieses Stück]
       ├─ Bellman-Ford + Floyd-Warshall → Johnson (Konvergenz: Umgewichtung) [nicht gebaut]
       └─ Mehrkriterien-Routing (Zeit gegen CO₂, Pareto)                    [nicht gebaut]
```

## Quellen

| Bestandteil | Quelle |
|---|---|
| Verfahren (Ordnen, Zusammenziehen mit Zeugensuche, Aufwärts-Abfrage) | Idee aus *Optimization Algorithms* (A. Khamis), Kap. 4.3.4, und der Arbeit von Geisberger, Sanders, Schultes und Delling (2008); Umsetzung, Ordnungen, Zeugengrenze und Auspacken sind **eigen** |
| Zahlen dieser Demo | **eigene Messungen** an eigenen Netzen, nicht die Zahlen der Quellen |
| **Kleines Netz**, Zufallsnetz, Stadtnetz | eigene Graphen und Erzeuger (Netze "in der Art des" Beispiels aus dem Buch) |
| **Toronto Innenstadt** | **echte OpenStreetMap-Daten**: befahrbares Netz im 9-km-Umkreis der City Hall (10 153 Knoten, 26 985 gerichtete Kanten), einmalig geholt (`tools/fetch_osm.py`), fest in `data/toronto_downtown.json`, Kosten in der App auf ganze Meter gerundet |

Aus den Büchern stammt nur die Idee; Text, Abbildungen, Code, Graphen und Zahlen der Bücher sind nicht übernommen.

**Daten und Lizenz:** Kartendaten © [OpenStreetMap-Mitwirkende](https://www.openstreetmap.org/copyright), Open Database License (ODbL) 1.0. `data/toronto_downtown.json` ist ein Auszug daraus und steht deshalb ebenfalls unter der ODbL – siehe [data/LICENSE-ODbL.md](data/LICENSE-ODbL.md).

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Kleines Netz (8 Orte) | ⚠️ Beim Zusammenziehen entstehen **6 gerichtete Abkürzungen** (drei Paare), an mehreren Stellen genügt ein Umweg als Zeuge – aber die CH-Abfrage legt **8** Knoten fest, die bidirektionale Suche 7: bei acht Orten lohnt sich das Vorrechnen nicht |
| Gewinn im echten Toronto-Netz | ✅ Median **21-fach** weniger festgelegte Knoten als bidirektional (Mittel über 30 Paare), für das gezeigte Paar **176** gegen 3 547 (bidirektional) und 6 093 (Dijkstra); dafür 37 340 Abkürzungen (**1.4 je Kante**), Vorberechnung etwa 3 Sekunden (Messwert) |
| Stadtnetz (Gitter) | ⚠️ im Mittel über fünf Netze nur **1.1-fach**; für das gezeigte Paar 51 statt 126 (bidirektional) und 241 (Dijkstra). Reichweite 1.0 / 1.5 / 2.3 / 3.2 → 3.3- / 1.9- / 1.1- / 0.9-fach; Größe 10 / 20 / 30 → 0.8- / 1.1- / 1.4-fach |
| Zufallsnetz | ❌ **0.5-fach**: die CH-Abfrage legt im Median fast doppelt so viele Knoten fest wie die bidirektionale Suche, die Vorberechnung fügt **1.9 Abkürzungen je Kante** ein; mittlerer Grad 2.5 / 4 / 6 → 1.2- / 0.6- / 0.4-fach |
| Ordnung der Knoten (Stadtnetz 16 × 16) | ⚠️ zufällige Ordnung **2.0** Abkürzungen je Kante gegen **0.6–0.7** bei Kantendifferenz; "nur Grad" braucht sogar weniger Abkürzungen (0.4), aber die Abfrage legt mehr Knoten fest. In Toronto lief die zufällige Ordnung in einem einzigen Versuch nach 280 Sekunden nicht durch – deshalb gibt es sie dort nicht zur Auswahl (Messung einmalig, nicht in den Tests) |
| Zeugengrenze (Toronto) | ✅ bei 1 / 5 / 20 / 200 Knoten je Zeugensuche **3.1 / 2.2 / 1.5 / 1.4** Abkürzungen je Kante – und **nie** falsche Kosten: weniger Zeugen kosten nur Platz |
| Verkehr ändert sich (Toronto) | ❌ schon bei **2 %** verdoppelten Kanten stimmen bei etwa **78 %** von 40 Paaren die berichteten Kosten nicht mehr und bei **60 %** ist die Route nicht mehr die kürzeste (im Mittel 2 % zu lang); bei 20 % sind es 100 % und 95 %. Neu vorrechnen (etwa 3 Sekunden) macht alle 40 Paare wieder exakt |
| Korrektheit | ✅ die CH-Abfrage liefert auf jedem geprüften Paar (alle Netze, alle Ordnungen, alle Zeugengrenzen) exakt die Kosten von einseitigem Dijkstra und networkx |

Die Zahl der festgelegten Knoten ist der Aufwand der Suche und plattformfest. Laufzeiten stehen in der App nur als Messwerte (reines Python); der Break-even – ab wie vielen Anfragen sich das Vorrechnen gegenüber Dijkstra ausgleicht – ist ein Messwert dieses Rechners und wird nirgends behauptet oder getestet.

## Was die Demo zeigt

1. **Vorberechnung: Knoten zusammenziehen.** Beim kleinen Netz ein Schritt-Regler durch die Reihenfolge mit den Entscheidungen je Nachbarpaar ("Zeuge gefunden" / "Abkürzung eingefügt", gestrichelt orange); bei den großen Netzen die Bilanz (Abkürzungen, Zeugensuchen, Vorrechenzeit) und die Abkürzungen je Rang.
2. **Die Abfrage in Aktion** (Schritt-Regler + Abspielen): CH vorwärts und rückwärts aufwärts in der Hierarchie, Treffpunkt (höchster Rang), ausgepackte Route; zuschaltbar die Flächen von Dijkstra (grau) und der bidirektionalen Suche.
3. **Erst vorrechnen, dann blitzschnell fragen:** Kennzahlen des gezeigten Paars mit Urteil, dazu die **Verteilung über 100 Zufallspaare** (Median, 10 %–90 %, Paare ohne Gewinn, falsche Kosten, Histogramm).
4. **Vergleich** (Expander) der Zähler und Messwerte; **Experimente auf Knopfdruck**: Netztypen, Ordnung der Knoten, Zeugengrenze, veraltete Kosten.
5. **Wo die Annahmen enden** (Tabelle) und **Mathematische Formulierung** (Invariante des Zusammenziehens, Beweisidee der Aufwärts-Abwärts-Route, Wichtigkeit).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz, Abstand Start–Ziel, **Ordnung der Knoten** und **Zeugensuche** wählen; die Adresszeile spiegelt die Konfiguration (Permalink). Regler, die zum gewählten Netz nicht gehören, sind ausgeblendet. Die Vorberechnung läuft beim ersten Aufruf eines Netzes live (Sekunden) und wird gemerkt.

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `ch_graph.py`, `ch_queues.py`, `ch_bd.py` | Graph in CSR-Form, Warteschlangen, einseitiges Dijkstra und bidirektionale Suche (Vergleichsverfahren aus dem vorigen Stück) |
| `ch_algorithm.py` | Vorberechnung (Ordnungen, Zeugensuche, Abkürzungen), CH-Abfrage, Auspacken, veraltete Kosten |
| `ch_scenario.py` | Netze: kleines Netz, Stadtnetz, Zufallsnetz, Toronto Innenstadt |
| `ch_evaluation.py` | Paarwahl, Kennzahlen, Verteilung über Paare, Experimente |
| `ch_visualization.py`, `ch_presets.py`, `ch_constants.py` | Abbildungen, Presets und Permalink, Konstanten |
| `tools/fetch_osm.py` | Einmal-Skript: holt das Toronto-Autonetz (braucht `osmnx`, nicht in `requirements.txt`) |
| `data/toronto_downtown.json` | der OSM-Auszug |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe läuft gegen networkx (`dijkstra_path_length`) für alle Ordnungen, gerichtete und ungerichtete Netze, Nullkanten, Parallelkanten und unerreichbare Paare.
