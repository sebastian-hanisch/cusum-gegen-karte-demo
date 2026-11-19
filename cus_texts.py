"""Texte der App: Einleitung, Expander, Annahmen, Formeln und die Tabellen als Markdown (aus den Rechenergebnissen gebildet, keine festen Befundzahlen).

Die Annahmentabelle steht auch in der README; tests/test_claims.py vergleicht beide Fassungen."""

import cus_arl as A
import cus_constants as C
import cus_handbeispiel as HB
from cus_evaluation import de, dg

INTRO = """
**Wer merkt einen Sprung im Mittelwert zuerst, wenn beide Kontrollkarten (kurz: Karten) ohne Sprung gleich oft falsch melden?** Die **Shewhart-Karte** urteilt aus der einen
Beobachtung (Alarm bei x > c), der **CUSUM** summiert die Abweichungen vom **Referenzmaß k**, solange sie anhalten: ein unbegrenztes Gedächtnis. Verglichen wird bei gleicher
**Fehlalarm-Lauflänge ARL₀** (Beobachtungen bis zum Alarm ohne Sprung) die **Verzögerung** (Beobachtungen bis zum Alarm nach dem Sprung; Lorden-Maß, hier der Wert bei ν = 0).
Der Preis des Gedächtnisses ist eine neue Annahme: **k kodiert, wie groß man sich den Sprung vorstellt.**

Die Reihen sind ausgedacht (unabhängige Normalzahlen mit bekannter Streuung, ein Sprung des Mittelwerts bei bekannter Größe in der einen Richtung). Es gibt keine reale Datenquelle;
in den gelesenen Quellen steht keine Messung, ob eine Alarmregel in Logistik, Lager oder Fertigung bessere Entscheidungen liefert. Die Aussagen gelten für dieses Modell mit
bekanntem Ziel und bekannter Streuung; gezählt wird in Beobachtungen bis zum Alarm.

Weiter unten: der Expander **So funktioniert der CUSUM**, vier Experimente (🔬), ein Expander **Was hilft bei großen Sprüngen?**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""


def handbeispiel_markdown() -> str:
    """Das Handbeispiel des Expanders, in Brüchen gerechnet (cus_handbeispiel.py)."""
    l = HB.cusum_run_lengths(HB.P0)
    x1, x2 = (3, 3, 3, 0, 4), (0, 4, 3, 1)
    ts1, tc1 = HB.alarm_times(x1)
    ts2, tc2 = HB.alarm_times(x2)
    c_mittel, s_mittel = HB.arl_cusum(HB.P_MITTEL), HB.arl_shewhart(HB.P_MITTEL)
    c_gross, s_gross = HB.arl_cusum(HB.P_GROSS), HB.arl_shewhart(HB.P_GROSS)
    return f"""
**Ein kleines ausgedachtes Zählmodell, in Brüchen gerechnet** (Beobachtung x ∈ {{0, …, 4}}, k = 2, h = 3; keine Betriebsdaten): Ohne Änderung ist P(x = 4) = 1/20.
Die Shewhart-Karte meldet bei x = 4, ARL₀ = 1/p = {HB.arl_shewhart(HB.P0)}. Der CUSUM hat drei Zustände S ∈ {{0, 1, 2}} und die Restlauflängen
L(0) = {l[0]}, L(1) = {l[1]}, L(2) = {l[2]} (lineares System, exakt); **beide Karten haben ARL₀ = {HB.arl_cusum(HB.P0)}** (Wahl des Beispiels, kein Phänomen).

- Reihe x = {x1}: Pfad S = {tuple(HB.cusum_pfad(x1))}; der CUSUM meldet bei t = {tc1}, die Shewhart-Karte bei t = {ts1}.
- Gegenstück x = {x2}: Pfad S = {tuple(HB.cusum_pfad(x2))}; die Shewhart-Karte meldet bei t = {ts2}, der CUSUM bei t = {tc2}. **Eine Reihe zeigt einen Fall, keine Regel.**
- Mit einem mittleren Sprung (P(x = 4) = 1/10) braucht der CUSUM {c_mittel} = {de(float(c_mittel), 4)} Beobachtungen, die Shewhart-Karte {s_mittel}; mit einem großen Sprung
  (P(x = 4) = 13/20) der CUSUM {c_gross} = {de(float(c_gross), 4)}, die Shewhart-Karte {s_gross} = {de(float(s_gross), 4)}. Der CUSUM gewinnt den kleinen bis mittleren Sprung,
  die Shewhart-Karte den großen (beide bei gleicher ARL₀ = 20).
"""


FUNKTIONIERT = r"""
1. **Rekursion.** Der CUSUM startet bei $S_0 = 0$ und nimmt in jedem Schritt $S_t = \max(0,\ S_{t-1} + x_t - k)$: die Abweichung der Beobachtung vom **Referenzmaß k** wird aufgeschlagen,
   der Wert bleibt nie unter 0. Alarm, sobald $S_t \ge h$ (Schwelle h; k und h in Einheiten der Streuung σ).
2. **Maximum der Teilsummen.** Dieselbe Zahl ist $S_t = \max\bigl(0,\ \max_{\nu} \sum_{i > \nu} (x_i - k)\bigr)$: die größte Summe der letzten Abweichungen über alle möglichen Sprungzeitpunkte ν.
   Das ist das unbegrenzte Gedächtnis (die Shewhart-Karte sieht nur die letzte Beobachtung).
3. **Brücke zur Wartezeit.** Die Rekursion ist formal dieselbe wie die Lindley-Rekursion der Wartezeit in einem Ein-Schalter-System (Zuwachs = Bedienzeit minus Zwischenankunftszeit); der Alarm „S ≥ h“
   ist dann die erste Passage der Wartezeit über h (Baron/Malov, nur Abstract und Einleitung gelesen). Das ist Mathematik, keine Aussage darüber, ob Praktiker Wartezeiten so überwachen.
4. **Woher k = δ/2 kommt.** Für einen Sprung von 0 auf δ ist der Log-Likelihood-Quotient einer Beobachtung $\delta x - \delta^2/2 = \delta\,(x - \delta/2)$: der CUSUM mit $k = \delta/2$ summiert diese
   Quotienten mit Neustart bei 0 (laut Moustakides u. a. 2009 und Bartroff/Tartakovsky 2024, Quellenbericht). **Wer k wählt, wählt damit einen Sprung δ = 2k.**
"""

ASSUMPTIONS_MARKDOWN = """
| Annahme | Was hier gilt, und was nicht gemessen ist | Wer setzt an |
|---|---|---|
| **Ziel 0 und σ = 1 sind bekannt** (die Beobachtung ist standardisiert) | Alle Zahlen setzen das voraus. Geschätzte Parameter sind nicht gemessen. | Stück „Parameter schätzen“ der Linie |
| **Unabhängige, normalverteilte Beobachtungen** | Autokorrelation und Schiefe sind nicht gemessen; die k-Tabelle gilt nur unter dieser Annahme. | Stück „Abhängige Werte“ der Linie |
| **Der Sprung ist konstant, einseitig und liegt zu Beginn (ν = 0) oder bei ν** | Einseitig gerechnet; zweiseitig nur für k ∈ {0,25; 0,5; 0,75; 1} und ARL₀ ∈ {100; 370; 1000} (eingefroren). | – |
| **Das Ziel ARL₀ = 370 ist gesetzt** (die 3σ-Karte zweiseitig hat 370,398) | Schwellen h gehören zu einem Ziel: 4,0954 einseitig und 4,7738 zweiseitig gelten für 370, 4,0965 und 4,7749 für 370,398. | – |
| **Maß: zero-state, hier gleich dem Lorden-Maß** | Quasistationär, zyklisch (Shiryaev) und Anfangsvorsprung sind nicht Stoff dieser Seite. | Stück „EWMA und das Maß“ der Linie |
| **Kein Betriebsbeleg** | Ob eine Alarmregel in Logistik, Lager oder Fertigung bessere Entscheidungen liefert, ist hier nicht gemessen. | – |
"""

MATH = r"""
**Lesehilfe.** Gegeben sind eine Reihe von Beobachtungen $x_1, x_2, \dots$ und zwei Alarmregeln; gesucht ist, wie viele Beobachtungen bis zum Alarm vergehen, ohne Sprung (ARL₀) und nach dem Sprung (ARL₁, Verzögerung).

| Symbol | Bedeutung |
|---|---|
| $x_t$ | Beobachtung $t = 1, 2, \dots$ (standardisiert: Ziel 0, Streuung 1) |
| $\delta$ | Sprung des Mittelwerts in Einheiten der Streuung |
| $\nu$ | Zahl der Beobachtungen vor dem Sprung (Beobachtung $\nu$ ist die letzte davor); die Alarmbeobachtung zählt mit, die Verzögerung ist $T - \nu \ge 1$ |
| $c$ | Grenze der Shewhart-Karte |
| $k$, $h$ | Referenzmaß und Schwelle des CUSUM |
| $S_t$ | CUSUM-Wert nach Beobachtung $t$ |
| $L(u)$ | mittlere Zahl der Beobachtungen bis zum Alarm, wenn der CUSUM bei $S = u$ steht |
| $\Phi$, $\varphi$ | Verteilungsfunktion und Dichte der Standardnormalverteilung |

**Modell.** Die Beobachtungen sind unabhängig, $x_t \sim N(0, 1)$ für $t \le \nu$ und $x_t \sim N(\delta, 1)$ für $t > \nu$. Ziel 0 und Streuung 1 sind bekannt.

**Shewhart-Karte (einseitig).** Alarm bei $x_t > c$. Die Lauflänge ist geometrisch mit dem Erfolgswert $p = 1 - \Phi(c - \delta)$:

$$\mathrm{ARL} = \frac{1}{1 - \Phi(c - \delta)}$$

Für eine vorgegebene ARL₀ ist $c = \Phi^{-1}(1 - 1/\mathrm{ARL}_0)$. Zweiseitig (Alarm bei $|x_t| > c$) ist $p = 1 - \Phi(c - \delta) + \Phi(-c - \delta)$ und $c = \Phi^{-1}(1 - 1/(2\,\mathrm{ARL}_0))$.

**CUSUM (einseitig, nach oben).** Mit $S_0 = 0$:

$$S_t = \max(0,\ S_{t-1} + x_t - k), \qquad T = \min\{t : S_t \ge h\}$$

Zweiseitig laufen zwei Statistiken, $S^{+}$ mit $x_t$ und $S^{-}_t = \max(0, S^{-}_{t-1} - x_t - k)$ mit $-x_t$; Alarm, sobald eine von beiden $\ge h$ ist. Mit zusätzlicher Shewhart-Grenze (Hybrid) meldet die Karte auch bei $x_t > c$.

**ARL des CUSUM aus der Integralgleichung.** Für $0 \le u < h$ gilt (Alarm, wenn der neue Wert $\ge h$ ist; der neue Wert ist 0 mit der Wahrscheinlichkeit $\Phi(k - u - \delta)$ und sonst stetig verteilt):

$$L(u) = 1 + L(0)\,\Phi(k - u - \delta) + \int_0^h L(y)\,\varphi(y - u + k - \delta)\,dy$$

Die ARL ist $L(0)$ (zero-state, Start bei $S_0 = 0$); mit $\delta = 0$ ist es die ARL₀, mit dem Sprung die Verzögerung ARL₁ bei $\nu = 0$. Die App löst die Gleichung mit dem Nyström-Verfahren (Gauß-Legendre-Knoten) und stimmt die Schwelle h durch Nullstellensuche auf die ARL₀ ab.

**Gitter (zweiter, unabhängiger Weg).** Brook/Evans-Kette: Zellbreite $w = h / (m - \tfrac12)$, Zustände $0, \dots, m-1$ (Zelle 0 ist $[0, w/2)$, Zelle $j$ ist $[(j - \tfrac12) w,\ (j + \tfrac12) w)$), Zellmitte $v_i = i\,w$:

$$R_{ij} = \Phi\!\left(\frac{(j + \tfrac12) w - v_i + k - \delta}{1}\right) - \Phi\!\left(\frac{(j - \tfrac12) w - v_i + k - \delta}{1}\right) \ (j \ge 1), \qquad R_{i0} = \Phi\!\left(\tfrac{w}{2} - v_i + k - \delta\right)$$

$$L = (I - R)^{-1}\,\mathbf{1}$$

Der Fehler ist proportional zu $1/m^2$ (Extrapolation nach Richardson aus zwei Zellzahlen). Die exakte Lauflängenverteilung folgt aus $P(T = t) = e_0^{\top} R^{t-1} (\mathbf{1} - R\,\mathbf{1})$.

**Maße der Verzögerung.** Für beide Karten ist die zero-state-Verzögerung gleich der Lorden-Verzögerung: Die Shewhart-Karte ist gedächtnislos (jedes $\nu$ gibt dieselbe Verzögerung), und beim CUSUM ist $\nu = 0$ der schlechteste Fall, weil die Restlauflänge $L(u)$ in $u$ nicht steigt (Moustakides u. a. 2009, laut Quellenbericht; hier zusätzlich am Gitter gemessen).
Quasistationär heißt der Grenzwert der Verzögerung für $\nu \to \infty$, bedingt auf „kein Alarm davor“; das zyklische Maß heißt Shiryaev-Maß (Stück „EWMA und das Maß“).

**Näherung für lange ARL₀.** Mit der Kullback-Leibler-Information $I = \delta^2/2$ gilt $J(\gamma) \sim \ln\gamma / I$ für $\gamma \to \infty$ (Lorden 1971 laut Bartroff/Tartakovsky 2024, nur Abstract gelesen); für endliches $\gamma$ ist es weder obere noch untere Schranke.

**Konvention ein- und zweiseitig.**

| Größe | einseitig | zweiseitig |
|---|---|---|
| CUSUM mit $k = 0{,}5$ für ARL₀ = 370 | $h = 4{,}0954$ | $h = 4{,}7738$ |
| Shewhart-Grenze für ARL₀ = 370 | $c = 2{,}7818$ | $c = 2{,}9997$ |
| ARL₀ des CUSUM mit $h = 4{,}77$ | 737,12 | rund 370 |

Eine Zahl „$h = 4{,}77$“ ohne Zusatz ist nur zweiseitig richtig. Die ARL₀ des zweiseitigen CUSUM ist bei $\delta = 0$ gleich der halben einseitigen (beobachtet, auf einem Gitter der Beobachtung gleich auf 2·10⁻¹⁵; nicht bewiesen).

**Literatur (Stufe: W = gelesen, sonst nicht gelesen).** Moustakides, Polunchenko, Tartakovsky 2009 (arXiv 0908.4119; W; ihre SADD-Spalten sind hier nicht zitiert). Moustakides 2008 (arXiv 0804.0741; W, Einleitung und Abschnitt 3).
Bartroff, Tartakovsky 2024 (arXiv 2403.18782; W, Satz 6). Lucas, Saccucci 1990 (Technometrics 32; W, Anhang A und Abschnitt 5; Lucas 1982 selbst nicht gelesen). Baron, Malov (arXiv 2502.09185; W, Abstract und Einleitung).
Zwetsloot, Jones-Farmer, Woodall 2024 (Quality Engineering 36; W). Nicht gelesen: Page 1954, Brook/Evans 1972, Lorden 1971 (nur Abstract), Moustakides 1986, Lucas 1982, Montgomery, Hawkins/Olwell.
Keine Neuheit: alles ist Lehrbuchstoff; neu ist nur die Vorführbarkeit mit Zähleinheit, zwei exakten Wegen und der k-Abhängigkeit.
"""


# ----------------------------------------------------------------------------- Tabellen als Markdown
def table_exp2(e) -> str:
    kopf = f"| δ | Shewhart | CUSUM k = {dg(e.k)} | CUSUM k = δ/2 | Verhältnis CUSUM k / Shewhart | Verhältnis CUSUM k = δ/2 / Shewhart |\n|---|---|---|---|---|---|\n"
    zeilen = []
    for d, s, ck, eh, rk, rh in e.table:
        zeilen.append(f"| {dg(d)} | {de(s, 3)} | {de(ck, 3)} | {'–' if eh is None else de(eh, 3)} | {de(rk, 3)} | {'–' if rh is None else de(rh, 3)} |")
    return kopf + "\n".join(zeilen)


def table_exp3(e) -> str:
    kopf = "| k \\ wahres δ | " + " | ".join(dg(d) for d in e.table_deltas) + " |\n|---|" + "---|" * len(e.table_deltas) + "\n"
    zeilen = []
    for k, row in zip(e.table_ks, e.table):
        cells = []
        for d, v in zip(e.table_deltas, row):
            txt = "–" if v is None else de(v, 3)
            if v is not None and abs(k - e.k) < 1e-9 and abs(d - e.delta) < 1e-9:
                txt = f"**{txt}**"
            cells.append(txt)
        zeilen.append(f"| {dg(k)} | " + " | ".join(cells) + " |")
    zeilen.append("| Shewhart | " + " | ".join(de(v, 3) for v in e.table_shewhart) + " |")
    zeilen.append("| bestes k* | " + " | ".join(de(v, 3) for v in e.k_opt_by_delta) + " |")
    return kopf + "\n".join(zeilen)


def table_exp4(e) -> str:
    kopf = "| γ = ARL₀ | h | J(γ) exakt | ln γ / I | Verhältnis J / (ln γ / I) | Differenz J − ln γ / I | Shewhart-Verzögerung |\n|---|---|---|---|---|---|---|\n"
    zeilen = []
    for r in e.rows:
        g = f"{r['gamma']:,}".replace(",", " ")
        txt = f"| {g} | {de(r['h'], 4)} | {de(r['J'], 4)} | {de(r['approx'], 4)} | {de(r['J'] / r['approx'], 4)} | {de(r['J'] - r['approx'], 3)} | {de(r['shewhart'], 3)} |"
        if r["gamma"] == e.gamma:
            txt = txt.replace("| " + g + " |", f"| **{g}** |", 1)
        zeilen.append(txt)
    return kopf + "\n".join(zeilen)


def table_hybrid(t: dict) -> str:
    deltas = t["deltas"]
    kopf = "| Zusatzgrenze c | h (ARL₀ = 370) | " + " | ".join(f"δ = {de(d, 1)}" for d in deltas) + " |\n|---|---|" + "---|" * len(deltas) + "\n"
    zeilen = []
    for z in t["zeilen"]:
        name = "∞ (reiner CUSUM)" if z["c"] is None else de(z["c"], 1)
        zeilen.append(f"| {name} | {de(z['h'], 5)} | " + " | ".join(de(v, 3) for v in z["arl1"]) + " |")
    zeilen.append("| Shewhart allein | – | " + " | ".join(de(v, 3) for v in t["shewhart_allein"]) + " |")
    return kopf + "\n".join(zeilen)


def streams_note() -> str:
    return ("Zufall: SplitMix64 mit Inversionsverfahren (`statistics.NormalDist().inv_cdf`), derselbe Zufalls-Seed gibt dieselbe Reihe; die simulierten Zahlen sind Zufallsergebnisse "
            f"(Mittel ± Standardfehler), die exakten Zahlen kommen aus der Kette. Rechenbudget (Planungswert): rund {C.SIM_BUDGET:,} Normalzahlen je Simulation.".replace(",", " "))


def table_umschlag(tab: dict, ks=C.TABLE_KS) -> str:
    kopf = "| ARL₀ (einseitige Grenze c) | " + " | ".join(f"k = {dg(k)}" for k in ks) + " |\n|---|" + "---|" * len(ks) + "\n"
    zeilen = []
    for arl0, werte in tab.items():
        c = A.shewhart_c(arl0)
        zeilen.append(f"| {arl0} (c = {de(c, 4)}) | " + " | ".join("–" if v is None else de(v, 4) for v in werte) + " |")
    return kopf + "\n".join(zeilen)
