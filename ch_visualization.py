"""Plotly-Abbildungen: Zusammenziehen des kleinen Netzes, Abfrage (Flächen der drei Verfahren), bilanz der Vorberechnung, Verteilungen, Experimente. Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import ch_constants as C

NET_NAMES = {"small": "Kleines Netz", "city": "Stadtnetz", "toronto": "Toronto Innenstadt", "random": "Zufallsnetz"}


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _edge_segments(g):
    """Alle Kanten als eine Linienspur (None trennt die Segmente); Hin- und Rückrichtung nur einmal."""
    src = np.repeat(np.arange(g.n), g.degree())
    dst = g.indices
    lo, hi = np.minimum(src, dst), np.maximum(src, dst)
    keep = np.zeros(len(src), dtype=bool)
    _, first = np.unique(lo * g.n + hi, return_index=True)
    keep[first] = True
    u, v = lo[keep], hi[keep]
    x = np.full(3 * len(u), None, dtype=object)
    y = np.full(3 * len(u), None, dtype=object)
    x[0::3], x[1::3] = g.xy[u, 0], g.xy[v, 0]
    y[0::3], y[1::3] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def _pad_ranges(fig, g):
    lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
    pad = 0.14 * (hi - lo)
    fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + pad[0]])
    fig.update_yaxes(range=[lo[1] - 0.15 * (hi[1] - lo[1]), hi[1] + 0.15 * (hi[1] - lo[1])])


def build_contraction(net, h, k, height=420):
    """Das kleine Netz nach `k` Zusammenziehungen: schon zusammengezogene Knoten grau mit ihrem Rang, der gerade zusammengezogene orange umrandet; bis dahin eingefügte Abkürzungen gestrichelt orange (mit Kosten)."""
    g = net.graph
    fig = go.Figure()
    src = np.repeat(np.arange(g.n), g.degree())
    seen = set()
    for u, v, w in zip(src.tolist(), g.indices.tolist(), g.weight.tolist()):
        if (v, u) in seen:
            continue
        seen.add((u, v))
        gone = h.rank[u] < k or h.rank[v] < k
        fig.add_trace(go.Scatter(x=[g.xy[u, 0], g.xy[v, 0]], y=[g.xy[u, 1], g.xy[v, 1]], mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color="rgba(190,190,190,0.5)" if gone else "rgba(120,120,120,0.9)", width=1.5)))
        mid = (g.xy[u] + g.xy[v]) / 2
        fig.add_annotation(x=mid[0], y=mid[1], text=f"{w:g}", showarrow=False, font=dict(size=11, color="#bbb" if gone else "#555"), bgcolor="rgba(255,255,255,0.7)")
    drawn = set()
    for v, r, entries in h.log[:k]:
        for u, w, via, wit, inserted in entries:
            if inserted and (min(u, w), max(u, w)) not in drawn and h.rank[u] >= r and h.rank[w] >= r:
                drawn.add((min(u, w), max(u, w)))
                fig.add_trace(go.Scatter(x=[g.xy[u, 0], g.xy[w, 0]], y=[g.xy[u, 1], g.xy[w, 1]], mode="lines", showlegend=False, hoverinfo="skip", line=dict(color=C.COLORS["shortcut"], width=3, dash="dash")))
                mid = (g.xy[u] + g.xy[w]) / 2
                fig.add_annotation(x=mid[0], y=mid[1], text=f"{via:g}", showarrow=False, font=dict(size=12, color=C.COLORS["shortcut"]), bgcolor="rgba(255,255,255,0.85)")
    gone = np.where(h.rank < k)[0]
    rest = np.where(h.rank >= k)[0]
    current = h.order[k - 1] if k > 0 else None
    if len(rest):
        fig.add_trace(go.Scatter(x=g.xy[rest, 0], y=g.xy[rest, 1], mode="markers+text", showlegend=False, text=[g.names[i] for i in rest], textposition="top center", hoverinfo="skip",
                                 marker=dict(size=14, color="white", line=dict(color="#333", width=2))))
    if len(gone):
        fig.add_trace(go.Scatter(x=g.xy[gone, 0], y=g.xy[gone, 1], mode="markers+text", showlegend=False, text=[f"{g.names[i]} (Rang {int(h.rank[i]) + 1})" for i in gone], textposition="top center",
                                 hoverinfo="skip", marker=dict(size=12, color="#d0d0d0", line=dict(color="#aaa", width=1.5))))
    if current is not None:
        fig.add_trace(go.Scatter(x=[g.xy[current, 0]], y=[g.xy[current, 1]], mode="markers", name="gerade zusammengezogen", hoverinfo="skip", marker=dict(size=22, color="rgba(255,255,255,0)", line=dict(color=C.COLORS["shortcut"], width=4))))
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    _pad_ranges(fig, g)
    return _base(fig, height)


def build_query(net, a, k, show_uni=True, show_bi=True, height=520):
    """Die Abfrage nach `k` Festlegungen der CH-Suche: die Flächen von Dijkstra (grau) und bidirektionaler Suche (hellblau/hellgrün) als Hintergrund, die Knoten der CH-Abfrage kräftig (vorwärts blau, rückwärts grün),
    am Ende der Treffpunkt (Stern) und die ausgepackte Route (schwarz)."""
    g, ch = net.graph, a.ch
    small = bool(g.names)
    fig = go.Figure()
    if net.geometric:
        ex, ey = _edge_segments(g)
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="rgba(150,150,150,0.4)", width=1), hoverinfo="skip", showlegend=False))
    else:
        fig.add_trace(go.Scatter(x=g.xy[:, 0], y=g.xy[:, 1], mode="markers", marker=dict(size=3, color="rgba(170,170,170,0.5)"), hoverinfo="skip", showlegend=False))
    msize = 14 if small else (3 if g.n > 1500 else 6)
    if show_uni and a.uni.order:
        o = np.array(a.uni.order, dtype=int)
        fig.add_trace(go.Scatter(x=g.xy[o, 0], y=g.xy[o, 1], mode="markers", name=f"Dijkstra: {len(o):,} Knoten".replace(",", "."), hoverinfo="skip", marker=dict(size=msize, color="rgba(150,150,150,0.35)")))
    if show_bi:
        for nodes, col, nm in ((a.bi.order_f, "rgba(31,119,180,0.35)", "bidirektional vorwärts"), (a.bi.order_b, "rgba(44,160,44,0.35)", "bidirektional rückwärts")):
            if nodes:
                o = np.array(nodes, dtype=int)
                fig.add_trace(go.Scatter(x=g.xy[o, 0], y=g.xy[o, 1], mode="markers", name=nm, hoverinfo="skip", marker=dict(size=msize, color=col)))
    sides = ch.steps[:k]
    fw = np.array([n for sd, n in sides if sd == "f"], dtype=int)
    bw = np.array([n for sd, n in sides if sd == "b"], dtype=int)
    big = msize + (4 if small else 5)
    for nodes, col, nm in ((fw, "#08519c", "CH vorwärts (aufwärts)"), (bw, "#006d2c", "CH rückwärts (aufwärts)")):
        if len(nodes):
            fig.add_trace(go.Scatter(x=g.xy[nodes, 0], y=g.xy[nodes, 1], mode="markers+text" if small else "markers", name=nm, hoverinfo="skip",
                                     text=[g.names[i] for i in nodes] if small else None, textposition="top center", marker=dict(size=big, color=col, line=dict(color="white", width=1))))
    if k >= len(ch.steps) and ch.route:
        pts = g.xy[ch.route]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", name="Route (ausgepackt)", line=dict(color=C.COLORS["route"], width=4), hoverinfo="skip"))
        p = ch.peak
        fig.add_trace(go.Scatter(x=[g.xy[p, 0]], y=[g.xy[p, 1]], mode="markers", name="Treffpunkt (höchster Rang)", hoverinfo="skip", marker=dict(size=18, color=C.COLORS["peak"], symbol="star", line=dict(color="white", width=1.5))))
    for node, nm, color, sym in ((a.s, "Start", C.COLORS["start"], "diamond"), (a.t, "Ziel", C.COLORS["goal"], "square")):
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", name=nm, hoverinfo="skip", marker=dict(size=15, color=color, symbol=sym, line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:
        _pad_ranges(fig, g)
    return _base(fig, height)


def build_added(h, height=260):
    """Eingefügte Abkürzungen je Zusammenziehung, in Gruppen von je 1 % der Knoten (Mittel je Knoten)."""
    n = len(h.added)
    bins = max(1, n // 50)
    x = np.arange(0, n, bins)
    y = [float(np.mean(h.added[i:i + bins])) for i in x]
    fig = go.Figure(go.Bar(x=x + bins / 2, y=y, marker_color=C.COLORS["shortcut"], hovertemplate="Rang um %{x:.0f}: %{y:.2f} Abkürzungen je Knoten<extra></extra>"))
    fig.update_layout(xaxis_title="Rang beim Zusammenziehen (0 = zuerst)", yaxis_title="Abkürzungen je Knoten (Mittel)")
    return _base(fig, height)


def build_gain_hist(gain, height=300):
    g = np.clip(np.asarray(gain, dtype=float), 0.1, 1000)
    fig = go.Figure(go.Histogram(x=np.log10(g), marker_color=C.COLORS["ch"], opacity=0.85, nbinsx=30, hovertemplate="10^%{x:.1f}-fach: %{y} Paare<extra></extra>"))
    fig.add_vline(x=0.0, line=dict(color="black", dash="dot"), annotation_text="kein Gewinn", annotation_position="top left")
    fig.add_vline(x=float(np.log10(np.median(g))), line=dict(color=C.COLORS["bi"], dash="dash"), annotation_text="Median", annotation_position="top right")
    fig.update_layout(xaxis=dict(title="Gewinn gegen die bidirektionale Suche (festgelegte Knoten, Zehnerpotenz)", tickvals=[-1, 0, 1, 2, 3], ticktext=["0.1×", "1×", "10×", "100×", "1000×"]), yaxis_title="Start-Ziel-Paare")
    return _base(fig, height)


def build_order(rows, height=300):
    labels = {"edge_difference": "Kantendifferenz (Nachbarn)", "lazy_edge_difference": "Kantendifferenz (träge)", "degree": "nur Grad", "random": "zufällig"}
    fig = go.Figure(go.Bar(x=[labels[r["order"]] for r in rows], y=[r["shortcut_ratio"] for r in rows], marker_color=C.COLORS["shortcut"], text=[f"{r['shortcut_ratio']:.2f}" for r in rows], textposition="outside",
                           hovertemplate="%{x}: %{y:.2f} Abkürzungen je Kante<extra></extra>"))
    fig.update_layout(yaxis=dict(title="Abkürzungen je Originalkante", rangemode="tozero"))
    return _base(fig, height)


def build_net_comparison(rows, height=340):
    fig = go.Figure()
    for key, name, col in (("median_uni", "Dijkstra", C.COLORS["uni"]), ("median_bi", "bidirektional", C.COLORS["bi"]), ("median_ch", "CH-Abfrage", C.COLORS["ch"])):
        fig.add_trace(go.Bar(x=[NET_NAMES[r["net"]] for r in rows], y=[r[key] for r in rows], name=name, marker_color=col))
    fig.update_layout(barmode="group", yaxis=dict(title="festgelegte Knoten je Abfrage (Median)", type="log"))
    return _base(fig, height)


def build_witness(rows, height=300):
    x = [str(r["witness"]) for r in rows]
    fig = go.Figure(go.Bar(x=x, y=[r["shortcut_ratio"] for r in rows], marker_color=C.COLORS["shortcut"], text=[f"{r['shortcut_ratio']:.2f}" for r in rows], textposition="outside",
                           hovertemplate="Zeugensuche bis %{x} Knoten: %{y:.2f} Abkürzungen je Kante<extra></extra>"))
    fig.update_layout(xaxis_title="festgelegte Knoten je Zeugensuche (höchstens)", yaxis=dict(title="Abkürzungen je Originalkante", rangemode="tozero"))
    return _base(fig, height)


def build_stale(rows, height=300):
    x = [f"{r['fraction']:.0%} der Kanten" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[r["share_wrong_cost"] * 100 for r in rows], name="berichtete Kosten falsch", marker_color="#7f7f7f"))
    fig.add_trace(go.Bar(x=x, y=[r["share_suboptimal"] * 100 for r in rows], name="Route nicht mehr die kürzeste", marker_color=C.COLORS["ch"]))
    fig.update_layout(barmode="group", yaxis=dict(title="Start-Ziel-Paare [%]", range=[0, 105]), xaxis_title="Anteil der Kanten, deren Kosten sich verdoppeln")
    return _base(fig, height)
