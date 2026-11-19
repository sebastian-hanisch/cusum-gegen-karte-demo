"""CUSUM gegen Shewhart-Karte: wer merkt den Sprung zuerst? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück der Linie „Änderungserkennung“: bei gleicher Fehlalarm-Lauflänge ARL0 die Verzögerung nach einem Sprung im Mittelwert (Shewhart-Karte gegen CUSUM), exakt aus zwei
unabhängigen Rechenwegen (Integralgleichung, Markov-Gitter), und das Referenzmaß k als Annahme über die Größe des Sprungs. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # kleine dichte Lösungen sind mit mehreren BLAS-Threads um den Faktor 100 langsamer

import streamlit as st  # noqa: E402

import cus_arl as A  # noqa: E402
import cus_constants as C  # noqa: E402
import cus_evaluation as E  # noqa: E402
import cus_layout as L  # noqa: E402
import cus_presets as P  # noqa: E402
import cus_reference as REF  # noqa: E402
import cus_texts as T  # noqa: E402
import cus_visualization as V  # noqa: E402
import modul_frisch  # noqa: E402

modul_frisch.aktualisieren(__file__)      # geänderte Hilfsmodule nach einem Update neu laden (Cloud behält sie sonst bis zum Neustart)

st.set_page_config(page_title="CUSUM gegen Shewhart-Karte – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _exact1(delta, arl0, k):
    return E.exact1(delta, arl0, k)


@st.cache_data(show_spinner=False)
def _sim1(seed, runs, delta, arl0, k):
    ex = E.exact1(delta, arl0, k)
    return E.sim1(seed, runs, delta, arl0, k, ex.h, ex.c, ex.arl1_shewhart)


@st.cache_data(show_spinner=False)
def _exp2(zaehlung, k, arl0):
    return E.exp2(zaehlung, k, arl0)


@st.cache_data(show_spinner=False)
def _umschlag():
    return E.umschlag_tabelle()


@st.cache_data(show_spinner=False)
def _exp3(delta, k, arl0):
    return E.exp3(delta, k, arl0)


@st.cache_data(show_spinner=False)
def _exp4(delta, gamma):
    return E.exp4(delta, gamma)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in cus_layout.py).
st.html(L.kopf_css(L.dunkles_design()))
with st.container(key="kopf"):
    st.title("📈 CUSUM gegen Shewhart-Karte: wer merkt den Sprung zuerst?")
st.markdown(T.INTRO)

with st.expander("💬 So funktioniert der CUSUM", expanded=True):
    st.markdown(T.FUNKTIONIERT)
    st.markdown(T.handbeispiel_markdown())

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

P.load_permalink_settings()
P.init_session_state_defaults()

with st.sidebar:
    P.sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Sprung und Karten**")
    delta = st.slider("Sprung δ (in σ)", *P.bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Größe des Sprungs im Mittelwert, in Einheiten der Streuung σ = 1. Gilt für Ergebnis, Experiment 1 und das wahre δ in Experiment 3.")
    arl0 = st.select_slider("Fehlalarm-Lauflänge ARL₀", options=C.ARL0_OPTIONS, key="arl0_select",
                            help="Beobachtungen bis zum Alarm ohne Sprung; beide Karten werden darauf abgestimmt (gesetztes Ziel, nicht 370,398).")
    k = st.slider("Referenzmaß k des CUSUM (in σ)", *P.bounds("k_slider"), step=C.K_STEP, format="%.3f", key="k_slider",
                  help="k kodiert, wie groß man sich den Sprung vorstellt: für bekanntes δ ist k = δ/2 das beste (bei auf ARL₀ abgestimmter Schwelle, Lorden-Maß). Die Schwelle h wird immer auf die gewählte ARL₀ abgestimmt.")
    st.button("k = δ/2 setzen", width="stretch", on_click=P.set_k_half_delta, key="k_half_button",
              help="Setzt k auf das Optimum für den gewählten Sprung δ (der Regler bleibt danach frei).")
    st.markdown("**Experiment 1: die Reihen**")
    nu = st.slider("Beobachtungen vor dem Sprung ν", *P.bounds("nu_slider"), key="nu_slider", help="Die Einzelreihe hat ν Beobachtungen ohne Sprung; die Reihen der Simulation haben den Sprung vor der ersten Beobachtung (ν = 0).")
    seed = st.number_input("Zufalls-Seed", *P.bounds("seed_input"), key="seed_input", step=1)
    reihen = st.slider("Zahl der Reihen R", *P.bounds("reihen_slider"), step=C.R_STEP, key="reihen_slider",
                       help="Simulierte Reihen mit Sprung (und ebenso viele ohne). Das Rechenbudget kappt R bei langen Verzögerungen; die App sagt es dann.")
    st.markdown("**Experiment 2: Vergleich über δ**")
    zaehlung = st.radio("Zählung", C.ZAEHLUNG, key="zaehlung_radio", horizontal=True,
                        help="Einseitig (nur Anstieg) oder zweiseitig (beide Richtungen). Zweiseitig gibt es nur eingefroren, für wenige k und ARL₀.")
    mass = st.radio("Maß der Anzeige", C.MASS, key="mass_radio", help="ARL₁ je Karte oder das Verhältnis CUSUM / Shewhart.")
    st.markdown("**Experiment 4: die Näherung ln γ / I**")
    ln_delta = st.select_slider("Sprung δ der Näherung", options=C.LN_DELTAS, key="ln_delta_select", help="Der CUSUM hat hier k = δ/2.")
    ln_gamma = st.select_slider("Hervorgehobenes γ", options=C.LN_GAMMAS, key="ln_gamma_select", help="Die Kurve zeigt alle acht γ, die Zeile in der Tabelle ist hervorgehoben.")
    st.button("🎲 Neuen Zufalls-Seed würfeln", width="stretch", on_click=P.randomize_seed)

P.sync_query_params({
    "delta_slider": delta, "arl0_select": arl0, "k_slider": k, "nu_slider": int(nu), "seed_input": int(seed), "reihen_slider": int(reihen),
    "zaehlung_radio": zaehlung, "mass_radio": mass, "ln_delta_select": ln_delta, "ln_gamma_select": ln_gamma,
})

settings = E.Settings(float(delta), int(arl0), float(k), int(nu), int(seed), int(reihen), zaehlung, mass, float(ln_delta), int(ln_gamma))

# --- Rechnungen (zwischengespeichert) ------------------------------------------------------------------------------------------------------
tunable = True
try:
    ex1 = _exact1(settings.delta, settings.arl0, settings.k)
except A.NotTunable as exc:
    tunable, tune_msg = False, str(exc)
e2 = _exp2(settings.zaehlung, settings.k, settings.arl0)
e3 = _exp3(settings.delta, settings.k, settings.arl0)
e4 = _exp4(settings.ln_delta, settings.ln_gamma)
e2_one = e2 if settings.zaehlung == C.ZAEHLUNG[0] else _exp2(C.ZAEHLUNG[0], settings.k, settings.arl0)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: eine Reihe, zwei Karten")
if not tunable:
    st.warning(f"{tune_msg}. Wählen Sie ein kleineres k oder eine größere ARL₀.")
else:
    m1, m2, m3, m4 = st.columns(4)
    # Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
    m1.metric("CUSUM: Verzögerung (exakt)", f"{E.de(ex1.arl1_cusum, 3)}", delta=f"{E.de(ex1.arl1_cusum - ex1.arl1_shewhart, 2)} gegen Shewhart", delta_color="inverse",
              help="Mittlere Zahl der Beobachtungen bis zum Alarm, wenn der Sprung vor der ersten Beobachtung liegt (zero-state, Lorden-Maß).")
    m2.metric("Shewhart-Karte: Verzögerung (exakt)", f"{E.de(ex1.arl1_shewhart, 3)}", delta="Referenz", delta_color="off")
    m3.metric("Verhältnis CUSUM / Shewhart", f"{E.de(ex1.arl1_cusum / ex1.arl1_shewhart, 3)}", delta="unter 1: der CUSUM ist schneller", delta_color="off")
    cross = e2_one.crossover if e2_one.available else None
    m4.metric("Umschlag δ* (einseitig, dieses k)", "–" if cross is None else E.de(cross, 4), delta="ab hier ist die Shewhart-Karte schneller" if cross is not None else "kein Umschlag im Raster",
              delta_color="off")
    series = E.default_series(settings, ex1.c, ex1.h)
    st.plotly_chart(V.build_series(series), width="stretch", key="series_chart")
    st.caption(E.series_caption(series.nu, series.t_shewhart, series.t_cusum))

# --- Kernfragen ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown(
    "1. **Wer meldet einen Sprung zuerst, wenn beide Karten gleich oft falsch melden?** " + (E.answer_q1(ex1) if tunable else "Hier nicht abstimmbar (siehe oben).") + "\n"
    "2. **Ab welcher Sprunggröße kippt es?** " + E.answer_q2(e2_one) + "\n"
    "3. **Was kostet ein falsch gewähltes k?** " + E.answer_q3(e3) + "\n"
    "4. **Wie nahe an der besten möglichen Verzögerung liegt der CUSUM?** " + E.answer_q4(e4)
)

st.markdown("---")

# --- Experiment 1 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: viele Reihen gegen die exakte Verteilung")
st.caption("R Reihen mit dem Sprung vor der ersten Beobachtung (ν = 0), beide Karten auf denselben Zahlen; dazu R Reihen ohne Sprung für den Anteil der Fehlalarme vor Beobachtung 100. "
           + T.streams_note())
if not tunable:
    st.warning("Ohne abgestimmte Schwelle gibt es keine Simulation; wählen Sie ein kleineres k oder eine größere ARL₀.")
else:
    wanted = settings.reihen
    runs = E.max_runs(ex1.arl1_shewhart, wanted)
    if runs < wanted:
        st.info(f"Zahl der Reihen auf {runs} gekappt (Rechenbudget, Planungswert: rund {C.SIM_BUDGET:,} Normalzahlen; die Shewhart-Verzögerung ist hier {E.de(ex1.arl1_shewhart, 1)} Beobachtungen).".replace(",", " "))
    t0 = time.perf_counter()
    with st.spinner(f"Simuliere {runs} Reihen mit und {runs} ohne Sprung (rund 1 bis 2 Sekunden, Zeit nur ein Hinweis) …"):
        sim = _sim1(settings.seed, wanted, settings.delta, settings.arl0, settings.k)
    elapsed = time.perf_counter() - t0
    tmax = E.display_tmax(ex1)
    width = E.bin_width(tmax)
    nb = -(-tmax // width)
    cus_sim = E.histogram(sim.delays_cusum, width, nb)
    shew_sim = E.histogram(sim.delays_shewhart, width, nb)
    cus_ex, shew_ex = E.exact_bins(ex1.pmf_cusum, ex1.p_shewhart, width, nb)
    st.plotly_chart(V.build_histogram(width, cus_sim, shew_sim, cus_ex, shew_ex), width="stretch", key="hist_chart")
    c1, c2, c3 = st.columns(3)
    c1.metric("CUSUM: Mittel in den Reihen", f"{E.de(sim.mean_cusum, 2)} ± {E.de(sim.se_cusum, 2)}", delta=f"exakt {E.de(ex1.arl1_cusum, 3)}", delta_color="off")
    c2.metric("Shewhart: Mittel in den Reihen", f"{E.de(sim.mean_shewhart, 2)} ± {E.de(sim.se_shewhart, 2)}", delta=f"exakt {E.de(ex1.arl1_shewhart, 3)}", delta_color="off")
    c3.metric(f"Fehlalarm bis Beobachtung {C.FALSE_ALARM_HORIZON} (CUSUM / Shewhart)", f"{sim.fa_cusum} / {sim.fa_shewhart} von {sim.runs}",
              delta=f"exakt {E.de(100 * ex1.fa_cusum)} % / {E.de(100 * ex1.fa_shewhart)} %", delta_color="off")
    kind, text = E.verdict_exp1(settings.delta, settings.arl0, settings.k, ex1, sim, settings.seed)
    getattr(st, kind)(text)
    st.caption(f"Rechenzeit dieser Simulation: {E.de(elapsed, 2)} s (bei zwischengespeicherter Wiederholung kürzer; Zeit nur ein Hinweis). In {sim.first_cusum} Reihen meldete der CUSUM zuerst, "
               f"in {sim.first_shewhart} die Shewhart-Karte, in {sim.tied} gleichzeitig.")

st.markdown("---")

# --- Experiment 2 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: gleiche ARL₀, wer ist schneller, und ab wann kippt es?")
st.caption(f"Verzögerung (ARL₁, Beobachtungen bis zum Alarm, zero-state) über den Sprung δ, beide Karten auf ARL₀ = {settings.arl0} abgestimmt, Zählung: {settings.zaehlung}. "
           "Einseitig live aus der Integralgleichung, zweiseitig nur aus der eingefrorenen Tabelle (2D-Kette).")
if e2.available:
    st.plotly_chart(V.build_arl_curves(e2, settings.mass), width="stretch", key="arl_chart")
    st.markdown(T.table_exp2(e2))
kind, text = E.verdict_exp2(e2)
getattr(st, kind)(text)
with st.expander("Umschlag δ* für andere k und ARL₀ (einseitig)"):
    st.caption("Der Sprung, ab dem die Shewhart-Karte bei gleicher ARL₀ die kleinere Verzögerung hat; es gibt keinen Umschlagpunkt „des CUSUM“, nur einen je k und ARL₀ (und je Zählung). Einseitig, zero-state.")
    st.markdown(T.table_umschlag(_umschlag()))

st.markdown("---")

# --- Experiment 3 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: welches k? Das Referenzmaß ist eine Annahme")
st.caption(f"Wahres δ = {E.de(settings.delta, 2)}, ARL₀ = {settings.arl0}, einseitig: ARL₁ über k, jedes k mit neu abgestimmter Schwelle h. Tabelle: Verhältnis zur besten Verzögerung je δ "
           "(Zeilen: festes k; Spalten: wahres δ).")
st.plotly_chart(V.build_k_curve(e3), width="stretch", key="k_chart")
st.markdown(T.table_exp3(e3))
kind, text = E.verdict_exp3(e3)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 --------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: wie schnell könnte man höchstens sein? Die Näherung ln γ / I")
st.caption(f"CUSUM mit k = δ/2, δ = {E.de(settings.ln_delta, 2)}, einseitig, zero-state: exakte Verzögerung J(γ) neben der Näherung ln γ / I (I = δ²/2) für acht γ von 10 bis 1 000 000.")
c_a, c_b = st.columns([3, 2])
c_a.plotly_chart(V.build_lorden(e4), width="stretch", key="lorden_chart")
c_b.plotly_chart(V.build_lorden_diff(e4), width="stretch", key="lorden_diff_chart")
st.markdown(T.table_exp4(e4))
kind, text = E.verdict_exp4(e4)
getattr(st, kind)(text)

st.markdown("---")

with st.expander("🛡️ Was hilft bei großen Sprüngen? CUSUM mit zusätzlicher Shewhart-Grenze"):
    hyb = REF.hybrid_table()
    st.caption(f"Eingefrorene Tabelle (Gitter, Richardson; Quelle: {hyb['_meta']['quelle']}, Stand {hyb['_meta']['stand']}): ARL₁ in Beobachtungen, k = 0,5, ARL₀ = 370 einseitig, "
               "h je Grenze c neu abgestimmt. Der Hinweis auf CUSUM mit Shewhart-Grenze steht in Lucas/Saccucci 1990 (Abschnitt 5, gelesen) als Verweis auf Lucas 1982, das nicht gelesen ist; die Zahlen sind eigene Nachrechnung.")
    st.markdown(T.table_hybrid(hyb))
    kind, text = E.verdict_hybrid(hyb)
    getattr(st, kind)(text)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(T.ASSUMPTIONS_MARKDOWN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(T.MATH)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Änderungserkennung: Alarm und Lauflänge](https://sebastianhanisch.net/konzepte-aenderungserkennung.html)."
)
