"""Kennzahlen, Experimente und Verdikte. Die Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`: die App ruft st.<Art>(Text),
tests/test_verdicts.py prüft Schwellen und jede Zahl im Text an handgesetzten Werten.

Modell: unabhängige Beobachtungen N(delta, 1), Ziel 0 und sigma = 1 bekannt; Zähleinheit Beobachtungen bis zum Alarm; Maß: zero-state (= Lorden-Maß
für beide Karten dieser Demo). Einseitig gerechnet, wenn nichts anderes dasteht. ARL0 = 370 ist das gesetzte Ziel (die 3-Sigma-Karte hat 370,398)."""

import math
from dataclasses import dataclass

import numpy as np

import cus_arl as A
import cus_chart as K
import cus_constants as C
import cus_reference as REF
from cus_scenario import make_series


# ----------------------------------------------------------------------------- Formatierung (Dezimalkomma)
def de(x: float, nd: int = 2) -> str:
    """Zahl mit Dezimalkomma; ab 10 000 mit schmalem geschütztem Leerzeichen als Tausendertrenner."""
    if x != x:  # nan (Standardfehler einer einzelnen Reihe)
        return "–"
    s = f"{x:,.{nd}f}"
    return s.replace(",", " ").replace(".", ",").replace("-", "−")


def dg(x: float, nd: int = 3) -> str:
    """Wie `de`, aber ohne überflüssige Nachkommastellen (δ = 1, k = 0,5, k = 0,125)."""
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return (s or "0").replace(".", ",")


def gam(g) -> str:
    """Ganzzahl (γ, ARL₀) mit schmalem geschütztem Leerzeichen als Tausendertrenner."""
    return f"{int(g):,}".replace(",", " ")


def fach(x: float, nd: int = 3) -> str:
    """„das 0,321fache“."""
    return f"{de(x, nd)}fache"


@dataclass(frozen=True)
class Settings:
    delta: float = C.DEFAULT_DELTA
    arl0: int = C.DEFAULT_ARL0
    k: float = C.DEFAULT_K
    nu: int = C.DEFAULT_NU
    seed: int = C.DEFAULT_SEED
    reihen: int = C.DEFAULT_R
    zaehlung: str = C.ZAEHLUNG[0]
    mass: str = C.MASS[0]
    ln_delta: float = C.DEFAULT_LN_DELTA
    ln_gamma: int = C.DEFAULT_LN_GAMMA


# ============================================================================= Experiment 1: eine Reihe, zwei Karten, viele Reihen
@dataclass(frozen=True)
class Exact1:
    """Exakte Größen der beiden Karten (einseitig, Sprung vor der ersten Beobachtung, zero-state)."""
    delta: float
    arl0: float
    k: float
    h: float
    c: float
    arl1_cusum: float
    arl1_shewhart: float
    median_cusum: int
    median_shewhart: int
    q95_cusum: int
    q95_shewhart: int
    fa_cusum: float          # P(Alarm vor Beobachtung 100 ... bis einschließlich 100) ohne Sprung, exakt aus der Kette
    fa_shewhart: float
    pmf_cusum: tuple         # P(Lauflänge = t), t = 0..T (nach dem Sprung)
    p_shewhart: float        # Alarmwahrscheinlichkeit je Beobachtung nach dem Sprung


def shewhart_quantile(p: float, q: float) -> int:
    """Kleinstes t mit P(Lauflänge <= t) = 1 - (1 - p)^t >= q (geometrische Verteilung)."""
    t = max(1, math.ceil(math.log1p(-q) / math.log1p(-p)))
    while 1.0 - (1.0 - p) ** t < q:
        t += 1
    while t > 1 and 1.0 - (1.0 - p) ** (t - 1) >= q:
        t -= 1
    return t


def exact1(delta: float, arl0: float, k: float) -> Exact1:
    h = A.cusum_tune(k, arl0, C.NODES)
    c = A.shewhart_c(arl0)
    p = float(A.ndtr(delta - c))
    pmf = A.length_distribution(A.grid_matrix(k, h, delta, C.GRID_M))
    pmf0 = A.length_distribution(A.grid_matrix(k, h, 0.0, C.GRID_M), tmax_cap=C.FALSE_ALARM_HORIZON + 1)
    return Exact1(
        delta=delta, arl0=arl0, k=k, h=h, c=c,
        arl1_cusum=A.cusum_arl(k, h, delta, C.NODES), arl1_shewhart=1.0 / p,
        median_cusum=A.quantile(pmf, 0.5), median_shewhart=shewhart_quantile(p, 0.5),
        q95_cusum=A.quantile(pmf, 0.95), q95_shewhart=shewhart_quantile(p, 0.95),
        fa_cusum=A.cdf_at(pmf0, C.FALSE_ALARM_HORIZON),
        fa_shewhart=1.0 - (1.0 - 1.0 / arl0) ** C.FALSE_ALARM_HORIZON,
        pmf_cusum=tuple(float(x) for x in pmf), p_shewhart=p,
    )


def max_runs(arl1_shewhart: float, wanted: int) -> int:
    """Obergrenze für die Zahl der Reihen aus dem Rechenbudget (Normalzahlen): je Reihe rund ARL1 der Shewhart-Karte plus 100 Beobachtungen ohne Sprung."""
    cap = int(C.SIM_BUDGET // (arl1_shewhart + C.FALSE_ALARM_HORIZON)) // 10 * 10
    return max(1, min(wanted, max(10, cap)))


@dataclass(frozen=True)
class Sim1:
    runs: int
    capped: bool
    mean_cusum: float
    se_cusum: float
    mean_shewhart: float
    se_shewhart: float
    first_cusum: int         # Reihen, in denen der CUSUM vor der Shewhart-Karte meldet
    first_shewhart: int
    tied: int
    fa_cusum: int            # Reihen ohne Sprung mit Alarm innerhalb der ersten 100 Beobachtungen
    fa_shewhart: int
    delays_cusum: tuple
    delays_shewhart: tuple
    normals: int


def sim1(seed: int, wanted_runs: int, delta: float, arl0: float, k: float, h: float, c: float, arl1_shewhart: float) -> Sim1:
    runs = max_runs(arl1_shewhart, wanted_runs)
    ts, tc, used1 = K.simulate_delays(seed + 10_000_019, runs, delta, c, k, h, C.SIM_TCAP)
    n_s, n_c, used2 = K.simulate_false_alarms(seed + 20_000_033, runs, C.FALSE_ALARM_HORIZON, c, k, h)
    m_c, se_c = K.mean_se(tc)
    m_s, se_s = K.mean_se(ts)
    return Sim1(
        runs=runs, capped=runs < wanted_runs, mean_cusum=m_c, se_cusum=se_c, mean_shewhart=m_s, se_shewhart=se_s,
        first_cusum=sum(a < b for a, b in zip(tc, ts)), first_shewhart=sum(a > b for a, b in zip(tc, ts)), tied=sum(a == b for a, b in zip(tc, ts)),
        fa_cusum=n_c, fa_shewhart=n_s, delays_cusum=tuple(tc), delays_shewhart=tuple(ts), normals=used1 + used2,
    )


def bin_width(tmax: int, max_bins: int = 40) -> int:
    return max(1, math.ceil(tmax / max_bins))


def histogram(delays, width: int, nbins: int):
    """Anteile der Läufe je Klasse [1 + i width, (i + 1) width] (Lauflänge t zählt von 1)."""
    n = len(delays)
    counts = [0] * nbins
    for t in delays:
        i = (t - 1) // width
        if i < nbins:
            counts[i] += 1
    return [c / n for c in counts]


def exact_bins(pmf_cusum, p_shewhart: float, width: int, nbins: int):
    """Exakte Wahrscheinlichkeiten je Klasse für beide Karten (CUSUM aus der Kette, Shewhart geometrisch)."""
    cus = [0.0] * nbins
    for t, pr in enumerate(pmf_cusum):
        if t >= 1:
            i = (t - 1) // width
            if i < nbins:
                cus[i] += pr
    shew = []
    for i in range(nbins):
        a, b = i * width, (i + 1) * width
        shew.append((1.0 - p_shewhart) ** a - (1.0 - p_shewhart) ** b)
    return cus, shew


def verdict_exp1(delta: float, arl0: int, k: float, ex: Exact1, sim: Sim1, seed: int) -> tuple[str, str]:
    """Verdikt Experiment 1: nennt die exakten Zahlen, die Zahlen dieser Reihen (Zufalls-Seed) und die Fehlalarm-Seite."""
    kind = "success" if ex.arl1_cusum < ex.arl1_shewhart else "warning"
    sim_c = f"{de(sim.mean_cusum)} ± {de(sim.se_cusum)}"
    sim_s = f"{de(sim.mean_shewhart)} ± {de(sim.se_shewhart)}"
    text = (
        f"δ = {dg(delta)}, ARL₀ = {arl0}, einseitig, k = {dg(k)}: exakt braucht der CUSUM im Mittel **{de(ex.arl1_cusum, 3)}** Beobachtungen bis zum Alarm "
        f"(Median {ex.median_cusum}, 95-%-Quantil {ex.q95_cusum}), die Shewhart-Karte **{de(ex.arl1_shewhart, 3)}** (Median {ex.median_shewhart}, 95-%-Quantil {ex.q95_shewhart}). "
        f"In diesen {sim.runs} Reihen (Zufalls-Seed {seed}) waren es {sim_c} gegen {sim_s} (Mittel ± Standardfehler). "
        f"Ohne Sprung melden beide Karten fast gleich oft falsch: bis Beobachtung {C.FALSE_ALARM_HORIZON} exakt {de(100 * ex.fa_cusum)} % (CUSUM) gegen {de(100 * ex.fa_shewhart)} % (Shewhart), "
        f"in den {sim.runs} Reihen ohne Sprung {sim.fa_cusum} gegen {sim.fa_shewhart}. Der Unterschied sitzt in der Verzögerung. "
        "Das Lorden-Maß ist für beide Karten der Wert bei ν = 0."
    )
    return kind, text


def display_tmax(ex: Exact1) -> int:
    """Obere Grenze der Anzeige des Histogramms: das größere der beiden 99-%-Quantile der Verzögerung."""
    return max(shewhart_quantile(ex.p_shewhart, 0.99), A.quantile(np.array(ex.pmf_cusum), 0.99))


def series_caption(nu: int, t_shewhart, t_cusum) -> str:
    """Satz unter der Einzelreihe: ein Fall, keine Regel; Fehlalarme vor dem Sprung und fehlende Alarme werden benannt."""
    def one(name, t):
        if t is None:
            return f"{name}: kein Alarm in {C.SERIES_LENGTH} Beobachtungen"
        if t <= nu:
            return f"{name}: Fehlalarm bei Beobachtung {t} (vor dem Sprung)"
        return f"{name}: Alarm bei Beobachtung {t}, Verzögerung {t - nu}"
    return f"Diese Reihe zeigt einen Fall, keine Regel. {one('Shewhart-Karte', t_shewhart)}; {one('CUSUM', t_cusum)}."


# ============================================================================= Experiment 2: gleiche ARL0, wer ist schneller
@dataclass(frozen=True)
class Exp2:
    zaehlung: str
    k: float
    arl0: int
    h: float | None
    c: float
    deltas: tuple
    shewhart: tuple
    cusum: tuple                  # ARL1 des CUSUM mit dem gewählten k (None, wo nicht verfügbar)
    envelope: tuple               # ARL1 mit k = delta/2 (None, wo nicht abstimmbar); nur einseitig
    table: tuple                  # Zeilen (delta, shewhart, cusum_k, cusum_halb, verhaeltnis_k, verhaeltnis_halb) auf TABLE_DELTAS
    crossover: float | None
    crossover_by_arl0: dict       # ARL0 -> Umschlag für das gewählte k (einseitig)
    available: bool
    note: str


def _safe_crossover(k: float, arl0: float):
    try:
        return A.crossover(k, arl0, C.NODES)
    except A.NotTunable:
        return None


def exp2_one_sided(k: float, arl0: int) -> Exp2:
    c = A.shewhart_c(arl0)
    try:
        h = A.cusum_tune(k, arl0, C.NODES)
    except A.NotTunable as exc:
        return Exp2("einseitig", k, arl0, None, c, (), (), (), (), (), None, {}, False, str(exc))
    deltas = C.ENVELOPE_DELTAS
    shew = tuple(A.shewhart_arl(c, d) for d in deltas)
    cus = tuple(A.cusum_arl(k, h, d, C.NODES) for d in deltas)

    def halb(d):
        try:
            return A.arl1_tuned(d / 2.0, arl0, d, C.NODES)
        except A.NotTunable:
            return None

    env = tuple(halb(d) for d in deltas)
    rows = []
    for d in C.TABLE_DELTAS:
        s = A.shewhart_arl(c, d)
        ck = A.cusum_arl(k, h, d, C.NODES)
        eh = halb(d)
        rows.append((d, s, ck, eh, ck / s, None if eh is None else eh / s))
    by = {a: _safe_crossover(k, a) for a in (100, 370, 1000)}
    return Exp2("einseitig", k, arl0, h, c, deltas, shew, cus, env, tuple(rows), _safe_crossover(k, arl0), by, True, "")


def exp2_two_sided(k: float, arl0: int) -> Exp2:
    cell = REF.twosided_cell(k, arl0)
    c = A.shewhart_c(arl0, two=True)
    if cell is None:
        return Exp2("zweiseitig", k, arl0, None, c, (), (), (), (), (), None, {}, False,
                    f"Zweiseitig ist nur für k ∈ {{{', '.join(de(x, 2) for x in C.TWOSIDED_KS)}}} und ARL₀ ∈ {{{', '.join(str(a) for a in C.TWOSIDED_ARL0)}}} eingefroren; "
                    f"für k = {de(k, 2)} und ARL₀ = {arl0} gibt es keine Zahl.")
    deltas = tuple(REF.twosided_deltas())
    shew = tuple(cell["arl1_shewhart"])
    cus = tuple(cell["arl1_cusum"])
    rows = tuple((d, s, ck, None, ck / s, None) for d, s, ck in zip(deltas, shew, cus))
    return Exp2("zweiseitig", k, arl0, cell["h"], cell["c_shewhart"], deltas, shew, cus, tuple(None for _ in deltas), rows,
                cell.get("umschlag"), {}, True, "")


def exp2(zaehlung: str, k: float, arl0: int) -> Exp2:
    return exp2_one_sided(k, arl0) if zaehlung == C.ZAEHLUNG[0] else exp2_two_sided(k, arl0)


def verdict_exp2(e: Exp2) -> tuple[str, str]:
    """Verdikt Experiment 2: Verhältnisse, Umschlag und die Zählung stehen in jedem Satz."""
    if not e.available:
        return "warning", e.note
    z = e.zaehlung
    by_delta = {row[0]: row for row in e.table}
    deltas = (0.75, 1.0, 4.0) if z == "einseitig" else (0.5, 1.0, 4.0)
    teile = [f"bei δ = {dg(d)} das {fach(by_delta[d][4])}" for d in deltas if d in by_delta]
    kopf = (f"{z.capitalize()}, ARL₀ = {e.arl0}, k = {dg(e.k)}, h = {de(e.h, 4)}: der CUSUM braucht " + ", ".join(teile[:-1]) + f" und {teile[-1]}"
            " der Shewhart-Verzögerung (Beobachtungen bis zum Alarm, zero-state).")
    if e.crossover is None:
        umschlag = " Einen Umschlag, ab dem die Shewhart-Karte schneller ist, gibt es in diesem Raster nicht."
    else:
        umschlag = f" Ab δ = {de(e.crossover, 4)} hat die Shewhart-Karte die kleinere Verzögerung (Umschlag)."
        if z == "einseitig":
            c100, c1000 = e.crossover_by_arl0.get(100), e.crossover_by_arl0.get(1000)
            if c100 is not None and c1000 is not None:
                umschlag += f" Der Umschlag wandert mit ARL₀: {de(c100, 2)} bei 100, {de(c1000, 2)} bei 1000 (gleiches k)."
    ende = ""
    if z == "einseitig":
        gerechnet = [row for row in e.table if row[3] is not None]
        n_win = sum(1 for row in gerechnet if row[3] < row[1])
        ende = f" Mit k = δ/2 gewinnt der CUSUM in {n_win} von {len(gerechnet)} geprüften δ."
        knapp = [row for row in gerechnet if row[3] < row[1] and row[5] >= 0.95]
        if knapp:
            row = knapp[-1]
            ende += f" Bei δ = {dg(row[0])} knapp: {de(row[3], 3)} gegen {de(row[1], 3)}."
    else:
        ende = " Das Verhältnis hängt am Zählen: Zahlen aus einseitiger und zweiseitiger Zählung dürfen nicht gemischt werden."
    return "info", kopf + umschlag + ende


# ============================================================================= Experiment 3: welches k
@dataclass(frozen=True)
class Exp3:
    delta: float
    k: float
    arl0: int
    ks: tuple                 # k-Raster der Kurve
    arl1: tuple               # ARL1(k) bei auf ARL0 abgestimmter Schwelle
    k_opt: float
    arl1_opt: float
    arl1_k: float | None      # ARL1 beim gewählten k (None, wenn nicht abstimmbar)
    ratio_k: float | None
    shewhart: float
    ratio_shewhart: float
    table_ks: tuple
    table_deltas: tuple
    table: tuple              # je k-Zeile: Verhältnisse ARL1(k)/ARL1* je delta (None, wo nicht abstimmbar)
    table_shewhart: tuple
    k_opt_by_delta: tuple
    arl1_opt_by_delta: tuple
    note: str


def exp3(delta: float, k: float, arl0: int) -> Exp3:
    c = A.shewhart_c(arl0)
    cap = min(C.K_MAX, c - 0.01)
    ks = sorted({round(0.05 * i, 2) for i in range(1, int(cap / 0.05) + 1)} | {round(delta / 2.0, 4)} | ({k} if k <= cap else set()))
    ks = [x for x in ks if x <= cap]
    arl1 = []
    for kk in ks:
        arl1.append(A.arl1_tuned(kk, arl0, delta, C.NODES))
    k_opt, a_opt = A.best_k(delta, arl0, C.NODES)
    try:
        a_k = A.arl1_tuned(k, arl0, delta, C.NODES)
        note = ""
    except A.NotTunable as exc:
        a_k, note = None, str(exc)
    shew = A.shewhart_arl(c, delta)
    k_by, a_by = [], []
    for d in C.TABLE_DELTAS:
        ko, ao = A.best_k(d, arl0, C.NODES)
        k_by.append(ko)
        a_by.append(ao)
    table = []
    for kk in C.TABLE_KS:
        try:
            h = A.cusum_tune(kk, arl0, C.NODES)
            table.append(tuple(A.cusum_arl(kk, h, d, C.NODES) / ao for d, ao in zip(C.TABLE_DELTAS, a_by)))
        except A.NotTunable:
            table.append(tuple(None for _ in C.TABLE_DELTAS))
    sh_row = tuple(A.shewhart_arl(c, d) / ao for d, ao in zip(C.TABLE_DELTAS, a_by))
    return Exp3(delta, k, arl0, tuple(ks), tuple(arl1), k_opt, a_opt, a_k, None if a_k is None else a_k / a_opt, shew, shew / a_opt,
                tuple(C.TABLE_KS), tuple(C.TABLE_DELTAS), tuple(table), sh_row, tuple(k_by), tuple(a_by), note)


def worst_ratio(row, deltas, max_delta: float = 3.0):
    """Größtes Verhältnis einer Tabellenzeile über die delta <= max_delta (None-Zellen zählen nicht)."""
    vals = [v for v, d in zip(row, deltas) if d <= max_delta and v is not None]
    return max(vals) if vals else None


def verdict_exp3(e: Exp3) -> tuple[str, str]:
    """Verdikt Experiment 3: der Preis eines falschen k in dieser Zelle, k = 0,5 und die Shewhart-Karte über alle geprüften delta."""
    if e.arl1_k is None:
        return "warning", e.note or "Für dieses k ist die ARL₀ nicht abstimmbar."
    row05 = e.table[e.table_ks.index(0.5)] if 0.5 in e.table_ks else None
    w05 = worst_ratio(row05, e.table_deltas) if row05 else None
    wsh = worst_ratio(e.table_shewhart, e.table_deltas)
    n_half = sum(1 for d, ko in zip(e.table_deltas, e.k_opt_by_delta) if abs(ko / d - 0.5) < 0.02)
    text = (
        f"Wahres δ = {dg(e.delta)}, k = {dg(e.k)} (für δ = {dg(2 * e.k)} gedacht), ARL₀ = {e.arl0}, einseitig: ARL₁ = **{de(e.arl1_k, 2)}** Beobachtungen gegen das Optimum "
        f"{de(e.arl1_opt, 2)} bei k = {dg(e.k_opt)}, das **{fach(e.ratio_k)}** (Lorden-Maß, zero-state). "
    )
    if w05 is not None:
        text += (f"k = 0,5 braucht bei den geprüften δ bis 3 höchstens das {fach(w05)} der jeweils besten Verzögerung, die Shewhart-Karte höchstens das {fach(wsh)}. ")
    text += (f"Das beste k ist δ/2 (in {n_half} von {len(e.table_deltas)} geprüften δ bei dieser ARL₀). Wer δ nicht kennt, kennt k nicht: "
             "„optimal“ heißt hier: für bekanntes δ, unabhängige Normalwerte mit bekanntem Ziel und σ, im Lorden-Maß.")
    return "info", text


# ============================================================================= Experiment 4: die Näherung ln(gamma) / I
@dataclass(frozen=True)
class Exp4:
    delta: float
    gamma: int
    rows: tuple     # je gamma: dict(gamma, h, J, approx, shewhart)


def exp4(delta: float, gamma: int) -> Exp4:
    rows = tuple(A.lorden_row(delta, g) for g in C.LN_GAMMAS)
    return Exp4(delta, gamma, rows)


def verdict_exp4(e: Exp4) -> tuple[str, str]:
    """Verdikt Experiment 4: über oder unter der exakten Verzögerung, gezählt über alle gerechneten gamma; keine Schranke in beiden Richtungen."""
    rows = e.rows
    ueber = [r for r in rows if r["approx"] > r["J"]]
    unter = [r for r in rows if r["approx"] < r["J"]]
    d0 = rows[0]["J"] - rows[0]["approx"]
    d1 = rows[-1]["J"] - rows[-1]["approx"]
    sel = next(r for r in rows if r["gamma"] == e.gamma)
    text = (
        f"δ = {dg(e.delta)}, k = δ/2 = {dg(e.delta / 2)}, einseitig, zero-state: die Näherung ln γ / I liegt bei {len(ueber)} von {len(rows)} geprüften γ über der exakten Verzögerung "
        f"J(γ) und bei {len(unter)} darunter. Bei γ = {gam(sel['gamma'])}: J = {de(sel['J'], 4)}, ln γ / I = {de(sel['approx'], 4)}, Differenz {de(sel['J'] - sel['approx'], 3)}. "
    )
    if unter and ueber:
        gammas = ", ".join(str(r["gamma"]) for r in unter)
        text += f"Darunter liegt sie bei γ = {gammas}: Die Näherung ist weder obere noch untere Schranke, nur eine Asymptotik für γ → ∞ im Verhältnis (Lorden 1971 laut Quellenbericht, nur Abstract gelesen). "
    elif ueber:
        text += f"Die Differenz J − ln γ / I geht von {de(d0, 2)} bei γ = {rows[0]['gamma']} auf {de(d1, 3)} bei γ = {gam(rows[-1]['gamma'])}; "
        text += "bei δ = 2 liegt sie für γ = 10 und 30 darunter (Regler „Sprung δ der Näherung“). "
    r_last = rows[-1]
    text += (f"Die Shewhart-Karte braucht bei γ = {gam(r_last['gamma'])}"
             + f" {de(r_last['shewhart'], 3)} Beobachtungen gegen {de(r_last['J'], 4)} beim CUSUM.")
    return "info", text


# ============================================================================= Tabelle: CUSUM mit Shewhart-Grenze
def verdict_hybrid(table: dict) -> tuple[str, str]:
    """Verdikt der Hybrid-Tabelle (eingefroren; k = 0,5, ARL₀ = 370 einseitig, h je Grenze c abgestimmt)."""
    deltas = table["deltas"]
    pure = next(z for z in table["zeilen"] if z["c"] is None)
    c35 = next(z for z in table["zeilen"] if z["c"] == 3.5)
    c30 = next(z for z in table["zeilen"] if z["c"] == 3.0)
    i05, i4 = deltas.index(0.5), deltas.index(4.0)
    r4 = c35["arl1"][i4] / pure["arl1"][i4]
    cost35 = c35["arl1"][i05] / pure["arl1"][i05] - 1.0
    cost30 = c30["arl1"][i05] / pure["arl1"][i05] - 1.0
    r4_30 = c30["arl1"][i4] / pure["arl1"][i4]
    text = (
        f"c = 3,5: bei δ = 4 auf das {fach(r4)} der reinen CUSUM-Verzögerung ({de(c35['arl1'][i4], 3)} gegen {de(pure['arl1'][i4], 3)}), bei δ = 0,5 um {de(100 * cost35, 1)} % schlechter "
        f"({de(c35['arl1'][i05], 3)} gegen {de(pure['arl1'][i05], 3)}); mit c = 3,0 kostet es {de(100 * cost30, 1)} % bei δ = 0,5 und bringt das {fach(r4_30)} bei δ = 4. "
        "Gilt für ARL₀ = 370 einseitig und k = 0,5; die Zusatzgrenze verkürzt die ARL₀ und wird deshalb in h ausgeglichen."
    )
    return "info", text


# ============================================================================= Hilfen für die App
def default_series(s: Settings, c: float, h: float):
    return make_series(s.seed, s.delta, s.nu, c, s.k, h)


# ============================================================================= Kernfragen (kurze Antworten, live)
def answer_q1(ex: Exact1) -> str:
    return (f"Bei δ = {dg(ex.delta)} und ARL₀ = {de(ex.arl0, 0)} (einseitig, k = {dg(ex.k)}) braucht der CUSUM im Mittel {de(ex.arl1_cusum, 3)} Beobachtungen bis zum Alarm, die Shewhart-Karte "
            f"{de(ex.arl1_shewhart, 3)}: das {fach(ex.arl1_cusum / ex.arl1_shewhart)} (exakt, Lorden-Maß). Experiment 1 zeigt es als Verteilung.")


def answer_q2(e: Exp2) -> str:
    if not e.available:
        return "Für diese Einstellung nicht abstimmbar (siehe Experiment 2)."
    if e.crossover is None:
        return f"Einseitig, ARL₀ = {e.arl0}, k = {dg(e.k)}: im gerechneten Raster (δ von 0,3 bis 7) kippt es nicht."
    return f"Einseitig, ARL₀ = {e.arl0}, k = {dg(e.k)}: ab δ = {de(e.crossover, 4)} hat die Shewhart-Karte die kleinere Verzögerung; der Umschlag hängt an ARL₀, k und an der Zählung."


def answer_q3(e: Exp3) -> str:
    if e.ratio_k is None:
        return "Für dieses k ist die ARL₀ nicht abstimmbar (siehe Experiment 3)."
    return (f"k = {dg(e.k)} (für δ = {dg(2 * e.k)} gedacht) bei wahrem δ = {dg(e.delta)}: das {fach(e.ratio_k)} der besten Verzögerung (bestes k* = {de(e.k_opt, 3)}, ARL₀ = {e.arl0}, einseitig).")


def answer_q4(e: Exp4) -> str:
    r = next(row for row in e.rows if row["gamma"] == e.gamma)
    rel = "über" if r["approx"] > r["J"] else "unter"
    return (f"Für δ = {dg(e.delta)} und γ = {gam(e.gamma)} liegt die Näherung ln γ / I = {de(r['approx'], 3)} {rel} der exakten Verzögerung J = {de(r['J'], 3)}; "
            "sie ist keine Schranke, nur eine Asymptotik für γ → ∞ im Verhältnis (Experiment 4).")


# ============================================================================= Umschlagpunkte über k und ARL0 (einseitig)
def umschlag_tabelle(arl0s=(100, 370, 1000), ks=C.TABLE_KS):
    """Umschlag δ* der einseitigen Shewhart-Karte gegen den CUSUM (bei gleicher ARL0) je ARL0 (Zeilen) und k (Spalten); None, wo es keinen gibt oder k nicht abstimmbar ist."""
    return {a: tuple(_safe_crossover(k, a) for k in ks) for a in arl0s}
