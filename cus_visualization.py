"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.

Senkrechte und waagerechte Hilfslinien sind Linienspuren mit Datenkoordinaten (kein add_vline/add_hline: auf logarithmischen Achsen liest Plotly deren Position anders)."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import cus_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _vline(fig, x, y0, y1, name, color, row=None, col=None, dash="dot"):
    trace = go.Scatter(x=[x, x], y=[y0, y1], mode="lines", line=dict(color=color, width=1.5, dash=dash), name=name, hoverinfo="skip")
    if row:
        fig.add_trace(trace, row=row, col=col or 1)
    else:
        fig.add_trace(trace)


# ----------------------------------------------------------------------------- Experiment 1
def build_series(series):
    """Einzelreihe: oben die Beobachtungen mit der Shewhart-Grenze, unten der CUSUM-Pfad mit der Schwelle h; die ersten Alarme sind markiert."""
    n = len(series.xs)
    t = list(range(1, n + 1))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.5, 0.5],
                        subplot_titles=("Beobachtungen und Shewhart-Grenze", "CUSUM-Pfad und Schwelle h"))
    fig.add_trace(go.Scatter(x=t, y=list(series.xs), mode="lines+markers", marker=dict(size=4, color=C.COLOR_GREY), line=dict(width=1, color=C.COLOR_GREY),
                             name="Beobachtung x", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1, n], y=[series.c, series.c], mode="lines", line=dict(color=C.COLOR_SHEWHART, width=2), name=f"Grenze c = {series.c:.3f}".replace(".", ","),
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=list(series.s_path), mode="lines", line=dict(color=C.COLOR_CUSUM, width=2), name="CUSUM S",
                             hovertemplate="t = %{x}: S = %{y:.2f}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[1, n], y=[series.h, series.h], mode="lines", line=dict(color=C.COLOR_CUSUM, width=2, dash="dash"), name=f"Schwelle h = {series.h:.3f}".replace(".", ","),
                             hoverinfo="skip"), row=2, col=1)
    ymax = max(max(series.xs), series.c, 1.0) + 0.5
    ymin = min(min(series.xs), -1.0) - 0.5
    smax = max(max(series.s_path), series.h) * 1.1 + 0.2
    if series.nu > 0:
        _vline(fig, series.nu + 0.5, ymin, ymax, "Sprung (ν = Beobachtungen davor)", "rgba(120,120,120,0.9)", row=1)
        _vline(fig, series.nu + 0.5, 0, smax, "Sprung", "rgba(120,120,120,0.9)", row=2)
        fig.data[-2].showlegend = True
        fig.data[-1].showlegend = False
    if series.t_shewhart is not None:
        _vline(fig, series.t_shewhart, ymin, ymax, f"Shewhart-Alarm bei t = {series.t_shewhart}", C.COLOR_SHEWHART, row=1, dash="solid")
    if series.t_cusum is not None:
        _vline(fig, series.t_cusum, 0, smax, f"CUSUM-Alarm bei t = {series.t_cusum}", C.COLOR_CUSUM, row=2, dash="solid")
    fig.update_yaxes(range=[ymin, ymax], title_text="x", row=1, col=1)
    fig.update_yaxes(range=[0, smax], title_text="S", row=2, col=1)
    fig.update_xaxes(title_text="Beobachtung t", row=2, col=1, range=[1, n])
    fig.update_xaxes(range=[1, n], row=1, col=1)
    return _base(fig, height=470)


def build_histogram(width, cus_sim, shew_sim, cus_exact, shew_exact):
    """Verzögerung nach dem Sprung (Lauflänge ab Beobachtung 1): Balken = simulierte Anteile, Linien = exakte Wahrscheinlichkeiten je Klasse."""
    nb = len(cus_sim)
    labels = [(i + 0.5) * width + 0.5 for i in range(nb)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=shew_sim, width=width * 0.42, offset=-width * 0.42, marker_color=C.COLOR_SHEWHART, opacity=0.6, name="Shewhart (simuliert)",
                         hovertemplate="Klasse um %{x:.0f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=cus_sim, width=width * 0.42, offset=0, marker_color=C.COLOR_CUSUM, opacity=0.6, name="CUSUM (simuliert)",
                         hovertemplate="Klasse um %{x:.0f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=shew_exact, mode="lines+markers", line=dict(color=C.COLOR_SHEWHART, width=2), marker=dict(size=5), name="Shewhart (exakt)",
                             hovertemplate="Klasse um %{x:.0f}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=labels, y=cus_exact, mode="lines+markers", line=dict(color=C.COLOR_CUSUM, width=2), marker=dict(size=5), name="CUSUM (exakt)",
                             hovertemplate="Klasse um %{x:.0f}: %{y:.3f}<extra></extra>"))
    label = "Verzögerung nach dem Sprung (Beobachtungen bis zum Alarm" + (f", Klassen der Breite {width})" if width > 1 else ")")
    fig.update_xaxes(title_text=label)
    fig.update_yaxes(title_text="Anteil der Läufe")
    fig.update_layout(barmode="overlay")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 2
def build_arl_curves(e, mass: str):
    """ARL1 (oder das Verhältnis zur Shewhart-Karte) über den Sprung δ; logarithmische Achsen, Hüllkurve k = δ/2 nur einseitig."""
    fig = go.Figure()
    d = list(e.deltas)
    ratio = mass == C.MASS[1]
    if ratio:
        y_c = [c / s for c, s in zip(e.cusum, e.shewhart)]
        fig.add_trace(go.Scatter(x=d, y=y_c, mode="lines+markers", line=dict(color=C.COLOR_CUSUM, width=3), name=f"CUSUM k = {e.k:g} / Shewhart".replace(".", ","),
                                 hovertemplate="δ = %{x}: %{y:.3f}<extra></extra>"))
        env = [(x, (v / s)) for x, v, s in zip(d, e.envelope, e.shewhart) if v is not None]
        if env:
            fig.add_trace(go.Scatter(x=[x for x, _ in env], y=[v for _, v in env], mode="lines", line=dict(color=C.COLOR_ENVELOPE, width=2, dash="dash"), name="CUSUM k = δ/2 / Shewhart",
                                     hovertemplate="δ = %{x}: %{y:.3f}<extra></extra>"))
        ymin, ymax = min(y_c + [v for _, v in env]) * 0.8, max(y_c + [1.0]) * 1.25
        fig.add_trace(go.Scatter(x=[d[0], d[-1]], y=[1, 1], mode="lines", line=dict(color=C.COLOR_SHEWHART, width=2), name="Shewhart-Karte (= 1)", hoverinfo="skip"))
        fig.update_yaxes(title_text="ARL₁ CUSUM / ARL₁ Shewhart (über 1: Shewhart schneller)", type="log", range=None)
    else:
        fig.add_trace(go.Scatter(x=d, y=list(e.shewhart), mode="lines+markers", line=dict(color=C.COLOR_SHEWHART, width=3), name="Shewhart-Karte",
                                 hovertemplate="δ = %{x}: %{y:.3f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=d, y=list(e.cusum), mode="lines+markers", line=dict(color=C.COLOR_CUSUM, width=3), name=f"CUSUM k = {e.k:g}".replace(".", ","),
                                 hovertemplate="δ = %{x}: %{y:.3f}<extra></extra>"))
        env = [(x, v) for x, v in zip(d, e.envelope) if v is not None]
        if env:
            fig.add_trace(go.Scatter(x=[x for x, _ in env], y=[v for _, v in env], mode="lines", line=dict(color=C.COLOR_ENVELOPE, width=2, dash="dash"), name="Hüllkurve: CUSUM mit k = δ/2",
                                     hovertemplate="δ = %{x}: %{y:.3f}<extra></extra>"))
        ymin = min(list(e.cusum) + [v for _, v in env]) * 0.8
        ymax = max(e.shewhart) * 1.25
        fig.update_yaxes(title_text="ARL₁ (Beobachtungen bis zum Alarm)", type="log")
    if e.crossover is not None:
        _vline(fig, e.crossover, ymin, ymax, f"Umschlag δ* = {e.crossover:.4f}".replace(".", ","), "rgba(120,120,120,0.9)")
    fig.update_xaxes(title_text="Sprung δ (in σ)", type="log")
    fig.update_yaxes(range=[_log10(ymin), _log10(ymax)])
    return _base(fig)


def _log10(x):
    import math
    return math.log10(max(x, 1e-9))


# ----------------------------------------------------------------------------- Experiment 3
def build_k_curve(e):
    """ARL1 über k bei auf ARL0 abgestimmter Schwelle: Minimum bei k = δ/2; die Shewhart-Karte als waagerechte Linie."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(e.ks), y=list(e.arl1), mode="lines+markers", line=dict(color=C.COLOR_CUSUM, width=3), marker=dict(size=5), name="CUSUM",
                             hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[e.ks[0], e.ks[-1]], y=[e.shewhart, e.shewhart], mode="lines", line=dict(color=C.COLOR_SHEWHART, width=2, dash="dash"), name="Shewhart-Karte",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[e.k_opt], y=[e.arl1_opt], mode="markers", marker=dict(size=13, symbol="star", color=C.COLOR_ENVELOPE), name=f"Optimum k* = {e.k_opt:.3f}".replace(".", ","),
                             hovertemplate="k* = %{x:.3f}: %{y:.3f}<extra></extra>"))
    if e.arl1_k is not None:
        fig.add_trace(go.Scatter(x=[e.k], y=[e.arl1_k], mode="markers", marker=dict(size=12, symbol="diamond-open", line=dict(width=3), color="black"), name="gewähltes k",
                                 hovertemplate="k = %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Referenzmaß k (in σ)")
    fig.update_yaxes(title_text="ARL₁ bei Sprung δ (Beobachtungen)", type="log")
    return _base(fig)


# ----------------------------------------------------------------------------- Experiment 4
def build_lorden(e):
    """J(γ) (exakt), ln γ / I (Näherung) und die Shewhart-Verzögerung über γ; logarithmische Achsen."""
    g = [r["gamma"] for r in e.rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g, y=[r["shewhart"] for r in e.rows], mode="lines+markers", line=dict(color=C.COLOR_SHEWHART, width=2), name="Shewhart-Karte",
                             hovertemplate="γ = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=g, y=[r["J"] for r in e.rows], mode="lines+markers", line=dict(color=C.COLOR_CUSUM, width=3), name="CUSUM exakt: J(γ)",
                             hovertemplate="γ = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=g, y=[r["approx"] for r in e.rows], mode="lines+markers", line=dict(color=C.COLOR_ENVELOPE, width=2, dash="dash"), name="Näherung ln γ / I",
                             hovertemplate="γ = %{x}: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Fehlalarm-Lauflänge γ = ARL₀", type="log")
    fig.update_yaxes(title_text="Verzögerung (Beobachtungen)", type="log")
    return _base(fig)


def build_lorden_diff(e):
    """Differenz J(γ) − ln γ / I: unter null liegt die Näherung über der exakten Verzögerung."""
    g = [r["gamma"] for r in e.rows]
    diff = [r["J"] - r["approx"] for r in e.rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g, y=diff, mode="lines+markers", line=dict(color=C.COLOR_CUSUM, width=3), name="J(γ) − ln γ / I",
                             hovertemplate="γ = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[g[0], g[-1]], y=[0, 0], mode="lines", line=dict(color="rgba(120,120,120,0.9)", width=1.5, dash="dot"), name="Differenz 0", hoverinfo="skip"))
    fig.update_xaxes(title_text="Fehlalarm-Lauflänge γ = ARL₀", type="log")
    fig.update_yaxes(title_text="Differenz (Beobachtungen)")
    return _base(fig, height=300)
