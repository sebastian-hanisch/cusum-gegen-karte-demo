# CUSUM gegen Shewhart-Karte: wer merkt den Sprung zuerst? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-cusum-gegen-karte-demo.streamlit.app/)**

Konzepte-Demo für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Stück 2 der Linie **Änderungserkennung** (Stück 1: „Karte und Lauflänge“, Stück 3: „EWMA und das Maß“).
Die Kante zum Vorgänger heißt `fix`: Die Alarmregel des Vorgängers urteilt aus **einer** Beobachtung (Shewhart-Karte) oder aus einem kurzen festen Fenster und braucht dafür keine Annahme über die Größe des Sprungs.
Dieses Stück gibt der Regel ein **unbegrenztes Gedächtnis** (der CUSUM summiert Abweichungen vom Referenzmaß k, solange sie anhalten). Der Preis ist **die neue Annahme**: k kodiert, wie groß man sich den Sprung vorstellt (für bekanntes δ ist k = δ/2 das beste, bei auf ARL₀ abgestimmter Schwelle, Lorden-Maß).
Verglichen wird bei gleicher Fehlalarm-Lauflänge, gezählt in **Beobachtungen bis zum Alarm** (die Alarmbeobachtung zählt mit). Die Demo ergänzt [`niveausprung-demo`](https://github.com/sebastian-hanisch/niveausprung-demo)
(dort läuft ein zweiseitiger CUSUM auf Regressionsresiduen eines Nachfrage-Erzeugers mit Prognose-Zurücksetzen, Bestätigungsregel und Kosten, ohne exaktes Orakel): **hier** gibt es die ARL₀ und ARL₁ **exakt auf zwei Wegen**,
die Wahl von k nach dem Sprung δ und den Vergleich mit der Shewhart-Karte bei gleicher ARL₀ auf der rohen Reihe. Die Reihen sind **ausgedacht** (Vehikel: unabhängige Normalzahlen mit bekannter Streuung und einem Sprung des Mittelwerts); es gibt keine reale Datenquelle.

## Warum dieses Problem

Eine Kontrollkarte (kurz: Karte) soll einen Sprung im Mittelwert melden, ohne bei unverändertem Prozess zu oft falsch zu melden. Beide Wünsche ziehen gegeneinander; fair vergleichen lässt sich nur bei **gleicher Fehlalarm-Lauflänge ARL₀**
(Beobachtungen bis zum Alarm ohne Sprung). Die Kernfragen der Demo: **Wer meldet einen Sprung zuerst, wenn beide Karten gleich oft falsch melden? Ab welcher Sprunggröße kippt es? Was kostet ein falsch gewähltes k? Wie nahe an der besten
möglichen Verzögerung liegt der CUSUM?** Die Demo sagt nichts darüber, ob eine Alarmregel in Logistik, Lager oder Fertigung bessere Entscheidungen liefert: In den gelesenen Quellen steht dazu keine Messung.

## Kernbotschaft

**Bei gleicher Fehlalarm-Lauflänge (ARL₀ = 370 Beobachtungen, einseitig; unabhängige Normalwerte mit bekanntem Ziel und σ; Lorden-Maß = zero-state) braucht der CUSUM mit k = 0,5 bei einem Sprung δ = 1 im Mittel 8,573 Beobachtungen bis zum Alarm,
die Shewhart-Karte 26,746; ab δ = 2,4375 kehrt sich das um (δ = 4: 1,744 gegen 1,126), und der Umschlag wandert mit ARL₀ (2,09 bei 100, 2,69 bei 1000), mit k (2,03 bei k = 0,25 bis 3,80 bei k = 1,5) und mit ein- oder zweiseitig (2,61 zweiseitig).
Das Referenzmaß k ist die Annahme über den Sprung: das beste k ist δ/2 (acht von acht gemessenen δ), ein falsches kostet auf diesem Raster (k ∈ {0,25 … 1,5} × δ ∈ {0,25 … 4}, ARL₀ = 370 einseitig) bis zum 2,99fachen der besten Verzögerung
(kein Satz über andere δ oder k), und die Näherung ln γ / I für die erreichbare Verzögerung ist keine Schranke (bei δ = 1 liegt sie bei allen geprüften γ darüber, bei δ = 2 und γ ≤ 30 darunter). Die Fehlalarm-Seite unterscheidet beide Karten kaum
(23,71 % gegen 23,04 % vor Beobachtung 100); der ganze Unterschied sitzt in der Verzögerung.** Gilt für unabhängige N(δ, 1)-Beobachtungen mit bekanntem Ziel und σ, Zähleinheit Beobachtungen.
Schwellen gehören zu einem Ziel: h = 4,0954 einseitig und h = 4,7738 zweiseitig für ARL₀ = 370 (die 3σ-Karte hat 370,398: 4,0965 und 4,7749); „h = 4,77“ ohne Zusatz ist nur zweiseitig richtig (einseitig gibt es die ARL₀ 737,12).

## Befunde und Korrekturen gegenüber dem Plan und den Vormessungen

Zahlen aus Plan und Vormessung waren Erwartungen; im Bau neu gerechnet (Integralgleichung, Gitter, 2D-Kette, SplitMix64 mit zwei Zufalls-Seeds). Fast alles stimmt auf die gedruckten Stellen, verschoben hat sich:

* **Umschlag zweiseitig:** der Plan nannte „≈ 2,61“ (lineare Interpolation auf dem 0,1-Raster); die Nullstelle der Kette (brentq) ist **2,6125** (k = 0,5, ARL₀ = 370). Die Tabelle unten nennt je k und ARL₀ die Nullstelle, nicht eine Interpolation.
* **Zweiseitige Shewhart-Zahlen:** die Vormessung stellte die Shewhart-Grenze auf die ARL₀ 369,9898 der Kette ein (155,0752 bei δ = 0,5), die eingefrorene Tabelle auf die Ziel-ARL₀ = 370 selbst (155,0790): Unterschied unter 6·10⁻⁴ relativ in den sieben Zellen des Plans.
  Die Schwelle ist dort ebenfalls auf 370 abgestimmt (h = 4,7738 gegen 4,7738 auf vier Stellen gleich).
* **Zufallsquelle und Rechenbudget:** der Plan rechnete mit dem Box–Muller-`normal()` (75,5 µs je Zahl, Budget 60 000 Normalzahlen). Die Simulation nimmt SplitMix64 mit dem Inversionsverfahren (`statistics.NormalDist().inv_cdf`):
  gemessen rund 1,2 µs je Zahl (37 424 Zahlen in 0,044 s, Zufalls-Seed 35, Standard), das Budget der App ist deshalb ein Planungswert von rund 1 000 000 Normalzahlen je Simulation (δ = 0,25 und ARL₀ = 5000: höchstens 470 Reihen; verbraucht werden dabei je nach Zufalls-Seed 1 010 466 bis 1 111 249, gemessen mit den Zufalls-Seeds 35 bis 37). Messung dieser README, nicht im Test.
* **Hälftenregel zweiseitig:** bei δ = 0 ist die zweiseitige ARL die Hälfte der einseitigen auf fünf Stellen (369,9898 gegen 369,9873, h = 4,7738); auf einem Gitter der Beobachtung gilt sie bis auf Rundungsfehler (relativ 10⁻¹¹ im Test; gemessen 2·10⁻¹⁵).
  Ob sie für stetige Normalwerte exakt gilt, ist nicht bewiesen; der Rest von 7·10⁻⁶ relativ ist Gitterfehler der 2D-Kette.
* **Gemischte Quellentabellen:** dieselbe Zelle (k = 0,5, h = 4, δ = 0,25) gibt zweiseitig 74,2241 und einseitig 77,0785, die Zelle (h = 5, δ = 0,5) 37,9962 und 38,0096. Eine Tabelle, die beide Zählungen mischt, ist die Namensfalle dieses Stücks; die App zeigt keine Quellentabellenzahl.
* **Hybrid-Tabelle:** h wird je Grenze c am Gitter m = 320 abgestimmt; der Richardson-Wert aus m = 320 und 640 liegt dort um den Gitterfehler daneben (ARL₀ 370,01 statt 370 bei c = 3,5).
* **Quasistationär:** die Verzögerung des CUSUM bei δ = 1 ist quasistationär 7,9038, zero-state 8,5730 (Lorden-Maß: der schlechteste Start ist S = 0). Das quasistationäre Maß heißt hier nie Shiryaev-Maß (zyklisch, nicht Stoff dieser Seite).
* **Regler k in Schritten von 0,025** (der Plan sagte 0,05 bis 2,5 ohne Schrittweite): so ist δ/2 für jedes δ des Reglers (Schritt 0,25) ein Wert des Reglers.

## Modell

Beobachtungen x_t, t = 1, 2, …, **unabhängig, N(0, 1) vor und N(δ, 1) ab der Änderung**; Ziel 0 und σ = 1 sind **bekannt**. **ν = Zahl der Beobachtungen vor der Änderung**; die Alarmbeobachtung zählt mit (Verzögerung T − ν ≥ 1).
**Shewhart-Karte** (einseitig): Alarm bei x_t > c. **CUSUM** (einseitig, nach oben): S_0 = 0, S_t = max(0, S_(t−1) + x_t − k), Alarm bei S_t ≥ h; k und h in σ-Einheiten. Zweiseitig: zwei solche Statistiken (mit x und mit −x).
**Zähleinheiten:** *Beobachtungen* (ARL und Verzögerung), *Läufe* (simulierte Reihen bis zum Alarm), *Zustände* (Zellen der Kette). „Fehlalarm“ heißt Alarm vor der Änderung, je Zeit gezählt (ARL₀), nicht der Anteil falscher unter den positiven.
**Maß:** für beide Karten ist die zero-state-Verzögerung gleich der Lorden-Verzögerung (die Shewhart-Karte ist gedächtnislos; beim CUSUM ist ν = 0 der schlechteste Fall, weil die Restlauflänge in S nicht steigt; am Gitter gemessen, nicht bewiesen).
Das Ziel **ARL₀ = 370 ist gesetzt** (nicht 370,398). Formeln, Integralgleichung, Gitter und Konventionstabelle ein-/zweiseitig stehen im Expander „📐 Mathematische Formulierung“ der App.

## Methodik

* **Einseitiger CUSUM auf zwei unabhängigen exakten Wegen:** Integralgleichung nach Nyström (120 Gauß-Legendre-Knoten, live in der App) und Brook/Evans-Gitter (Zellbreite w = h/(m − ½), Richardson aus zwei Zellzahlen; Test und exakte Lauflängenverteilung der App). Die Schwelle h wird durch Nullstellensuche (brentq) auf die ARL₀ abgestimmt.
* **Shewhart-Karte** geschlossen: ARL = 1/p. **Zweiseitiger CUSUM** als 2D-Kette (Richardson aus m = 60 und 80) und **CUSUM mit Shewhart-Grenze** am Gitter (Richardson aus m = 320 und 640): beide **eingefroren** (`cus_twosided.json`, `cus_hybrid.json`, Skript `tools/cus_make_tables.py`, rund 200 s,
  `--check` rechnet drei Zellen je Datei neu).
* **Simulation** nur in Experiment 1: SplitMix64, Inversionsverfahren, beide Karten auf denselben Zahlen; Mittel ± Standardfehler gegen die exakte Größe; der Anteil der Fehlalarme vor Beobachtung 100 aus Reihen ohne Sprung gegen die Kette.
* **Optimales k** durch Minimierung über k (jedes k mit neu abgestimmter Schwelle), **Umschlag δ\*** durch Nullstellensuche der Differenz CUSUM − Shewhart.

## Befunde (gemessen, keine Behauptungen)

Alle Tabellen sind Ausgaben der App-Funktionen mit den Standardwerten (ARL₀ = 370, einseitig, k = 0,5) und stehen in `tests/test_claims.py` Zelle für Zelle gegen die README (Marke `tabelle:<name>`); die Werte selbst sind in den Orakel-Tests gegen andere Rechenwege geprüft.

**Tabelle A: CUSUM gegen Shewhart-Karte bei gleicher ARL₀ = 370 (einseitig, zero-state; ARL₁ in Beobachtungen).** Mit k = δ/2 gewinnt der CUSUM in 8 von 8 geprüften δ (bei δ = 4 knapp: 1,121 gegen 1,126).

<!-- tabelle:vergleich -->
| δ | Shewhart | CUSUM k = 0,5 | CUSUM k = δ/2 | Verhältnis CUSUM k / Shewhart | Verhältnis CUSUM k = δ/2 / Shewhart |
|---|---|---|---|---|---|
| 0,25 | 176,257 | 81,866 | 57,377 | 0,464 | 0,326 |
| 0,5 | 88,890 | 27,674 | 23,668 | 0,311 | 0,266 |
| 0,75 | 47,426 | 13,640 | 13,224 | 0,288 | 0,279 |
| 1 | 26,746 | 8,573 | 8,573 | 0,321 | 0,321 |
| 1,5 | 10,005 | 4,843 | 4,572 | 0,484 | 0,457 |
| 2 | 4,605 | 3,406 | 2,917 | 0,740 | 0,633 |
| 3 | 1,705 | 2,231 | 1,571 | 1,308 | 0,921 |
| 4 | 1,126 | 1,744 | 1,121 | 1,550 | 0,996 |
<!-- /tabelle -->

**Tabelle B: Umschlag δ\*** (der Sprung, ab dem die einseitige Shewhart-Karte die kleinere Verzögerung hat), je ARL₀ und k. Der Umschlag steigt mit ARL₀ und mit k, aber langsamer als k (δ\*/k bei ARL₀ = 370: 8,1 / 4,9 / 3,7 / 3,1 / 2,5);
es gibt keinen Umschlagpunkt „des CUSUM“, nur einen je k, ARL₀ und Zählung. „Der CUSUM verliert ab δ = 2,45“ war die Zelle k = 0,5, ARL₀ = 370 auf einem 0,05-Raster; exakt ist 2,4375.

<!-- tabelle:umschlag -->
| ARL₀ (einseitige Grenze c) | k = 0,25 | k = 0,5 | k = 0,75 | k = 1 | k = 1,5 |
|---|---|---|---|---|---|
| 100 (c = 2,3263) | 1,7154 | 2,0914 | 2,4437 | 2,8107 | 3,5313 |
| 370 (c = 2,7818) | 2,0343 | 2,4375 | 2,7426 | 3,0767 | 3,8047 |
| 1000 (c = 3,0902) | 2,2676 | 2,6857 | 2,9767 | 3,2685 | 3,9811 |
<!-- /tabelle -->

**Tabelle C: zweiseitig (eingefrorene 2D-Kette; CUSUM k = 0,5, h = 4,7738, zweiseitige Shewhart-Grenze c = 2,9997, ARL₀ = 370).** Umschlag zweiseitig 2,6125, einseitig 2,4375. **Das Verhältnis hängt am Zählen:** zweiseitig 0,226 bei δ = 1, einseitig 0,321, weil die
zweiseitige Shewhart-Karte (c ≈ 3) bei δ = 1 deutlich langsamer ist (43,861 gegen 26,746), der zweiseitige CUSUM aber kaum (9,925 gegen 8,573). Ein Vergleich über beide Zählungen hinweg wäre eine Namensfalle.

<!-- tabelle:zweiseitig -->
| δ | Shewhart | CUSUM k = 0,5 | CUSUM k = δ/2 | Verhältnis CUSUM k / Shewhart | Verhältnis CUSUM k = δ/2 / Shewhart |
|---|---|---|---|---|---|
| 0,25 | 280,865 | 121,598 | – | 0,433 | – |
| 0,5 | 155,079 | 35,254 | – | 0,227 | – |
| 0,75 | 81,146 | 16,187 | – | 0,199 | – |
| 1 | 43,861 | 9,925 | – | 0,226 | – |
| 1,5 | 14,958 | 5,521 | – | 0,369 | – |
| 2 | 6,300 | 3,858 | – | 0,612 | – |
| 2,5 | 3,240 | 3,000 | – | 0,926 | – |
| 3 | 1,999 | 2,486 | – | 1,243 | – |
| 4 | 1,188 | 1,957 | – | 1,647 | – |
<!-- /tabelle -->

**Verteilung statt Mittel (δ = 1, ARL₀ = 370, einseitig, exakt aus der Kette).** Verzögerung nach dem Sprung: CUSUM Mittel 8,573, Median 7, 95-%-Quantil 18, P(Lauflänge ≤ 10) = 0,7386; Shewhart-Karte Mittel 26,746, Median 19, 95-%-Quantil 79.
Ohne Sprung melden beide Karten fast gleich oft falsch: **23,71 %** (Shewhart) gegen **23,04 %** (CUSUM) bis einschließlich Beobachtung 100, Median 257 gegen 258 (die Verteilung der Fehlalarm-Lauflänge ist nahezu geometrisch; der Anteil ist die Zahl, nicht die Gleichheit).

**Simulation (SplitMix64, 300 Reihen mit Sprung, 300 ohne; Standardwerte).** Zufalls-Seed 35: CUSUM 8,65 ± 0,25 gegen exakt 8,573, Shewhart 27,25 ± 1,48 gegen 26,746; Fehlalarm vor Beobachtung 100 in 56 (CUSUM) und 58 (Shewhart) von 300 Reihen, erwartet 69,1 und 71,1.
Zufalls-Seed 2026: 9,02 ± 0,28 und 29,30 ± 1,77, Fehlalarme 75 und 79. Über die Zufalls-Seeds 1 bis 40 liegt der Mittelwert des CUSUM in 40, der der Shewhart-Karte in 39 von 40 Fällen innerhalb von zwei Standardfehlern der exakten Größe (erwartet rund 38);
die Fehlalarm-Anteile in 38 (CUSUM) und 39 (Shewhart) von 40. Die Zahlen der Simulation sind Zufallsergebnisse (Band), die exakten Zahlen nicht. In 217 der 300 Reihen (Zufalls-Seed 35) meldete der CUSUM zuerst, in 48 die Shewhart-Karte, in 35 beide gleichzeitig: das ist ein Fall, keine Regel.

**Tabelle D: das beste k ist δ/2, und ein falsches k kostet.** Verhältnis ARL₁(k) / ARL₁(k\*) bei ARL₀ = 370 (Zeilen: festes k; Spalten: wahres δ; k\* je δ in der letzten Zeile). k\*/δ = 0,5 in allen acht Fällen (ARL₁\* = 57,377 / 23,668 / 13,224 / 8,573 / 4,572 / 2,917 / 1,571 / 1,121).
Das gilt **innerhalb der CUSUM-Familie** und für diese acht δ. Auf diesem Raster ist die größte Einbuße bei zu großem k (bei kleinem δ: bis 2,991) höher als bei zu kleinem k (bei großem δ: bis 2,043); die Abstände k/k\* sind dabei nicht gleich groß; k = 0,5 bleibt für δ von 0,25 bis 3 höchstens beim 1,427fachen, die Shewhart-Karte kommt bis zum 3,756fachen.
**Der CUSUM ist nicht robust gegen ein falsches k;** „optimal“ heißt: für bekanntes δ, unabhängige Normalwerte mit bekanntem Ziel und σ, im Lorden-Maß (Moustakides 1986, nur über Folgequellen gelesen).

<!-- tabelle:k -->
| k \ wahres δ | 0,25 | 0,5 | 0,75 | 1 | 1,5 | 2 | 3 | 4 |
|---|---|---|---|---|---|---|---|---|
| 0,25 | 1,080 | 1,000 | 1,043 | 1,128 | 1,330 | 1,534 | 1,914 | 2,043 |
| 0,5 | 1,427 | 1,169 | 1,031 | **1,000** | 1,059 | 1,168 | 1,420 | 1,556 |
| 0,75 | 1,805 | 1,532 | 1,239 | 1,081 | 1,000 | 1,036 | 1,192 | 1,214 |
| 1 | 2,167 | 1,996 | 1,593 | 1,301 | 1,047 | 1,000 | 1,066 | 1,077 |
| 1,5 | 2,749 | 2,991 | 2,568 | 2,062 | 1,391 | 1,112 | 1,000 | 1,009 |
| Shewhart | 3,072 | 3,756 | 3,586 | 3,120 | 2,188 | 1,579 | 1,086 | 1,004 |
| bestes k* | 0,125 | 0,250 | 0,375 | 0,500 | 0,750 | 1,000 | 1,500 | 2,000 |
<!-- /tabelle -->

**Tabelle E: die Näherung ln γ / I (I = δ²/2) gegen die exakte Verzögerung J(γ)** (einseitig, k = δ/2, zero-state; γ = ARL₀). Bei δ = 0,5 und δ = 1 liegt die Näherung bei allen acht geprüften γ **über** J; bei δ = 2 liegt sie bei γ = 10 und 30 **darunter**, ab γ = 100 darüber.
Sie ist also weder obere noch untere Schranke, sondern eine Asymptotik für γ → ∞ im Verhältnis (J / (ln γ / I) → 1; Lorden 1971 laut Bartroff/Tartakovsky 2024, nur Abstract gelesen), die Differenz bleibt bei einer Konstanten. Die Differenz ändert sich zwischen γ = 10⁵ und 10⁶ nur noch um 0,0064 bei δ = 0,5 (−24,6006 gegen −24,6070) und um 0,0005 bei δ = 1 (−3,3307 gegen −3,3311).
Die Shewhart-Karte braucht bei γ = 10⁶ und δ = 1 11 465,574 Beobachtungen gegen 24,2999 beim CUSUM (Verhältnis 471,8). Die SADD-Spalten von Moustakides u. a. 2009 liegen nach der Vormessung unter der exakten Rechnung und sind nicht zitiert.

<!-- tabelle:lorden05 -->
| γ = ARL₀ | h | J(γ) exakt | ln γ / I | Verhältnis J / (ln γ / I) | Differenz J − ln γ / I | Shewhart-Verzögerung |
|---|---|---|---|---|---|---|
| 10 | 1,3478 | 4,3462 | 18,4207 | 0,2359 | −14,075 | 4,603 |
| 30 | 2,6251 | 8,3855 | 27,2096 | 0,3082 | −18,824 | 10,975 |
| 100 | 4,4182 | 14,8451 | 36,8414 | 0,4029 | −21,996 | 29,499 |
| 370 | 6,7076 | 23,6681 | 47,3080 | 0,5003 | −23,640 | 88,890 |
| 1 000 | 8,5851 | 31,0829 | 55,2620 | 0,5625 | −24,179 | 208,526 |
| 10 000 | 13,1113 | 49,1331 | 73,6827 | 0,6668 | −24,550 | 1 554,834 |
| 100 000 | 17,7052 | 67,5028 | 92,1034 | 0,7329 | −24,601 | 12 003,225 |
| 1 000 000 | 22,3089 | 85,9171 | 110,5241 | 0,7774 | −24,607 | 95 000,088 |
<!-- /tabelle -->

<!-- tabelle:lorden10 -->
| γ = ARL₀ | h | J(γ) exakt | ln γ / I | Verhältnis J / (ln γ / I) | Differenz J − ln γ / I | Shewhart-Verzögerung |
|---|---|---|---|---|---|---|
| 10 | 0,9109 | 2,4920 | 4,6052 | 0,5411 | −2,113 | 2,570 |
| 30 | 1,7888 | 4,0436 | 6,8024 | 0,5944 | −2,759 | 4,946 |
| 100 | 2,8494 | 6,1078 | 9,2103 | 0,6631 | −3,103 | 10,827 |
| 370 | 4,0954 | 8,5730 | 11,8270 | 0,7249 | −3,254 | 26,746 |
| 1 000 | 5,0707 | 10,5171 | 13,8155 | 0,7613 | −3,298 | 54,649 |
| 10 000 | 7,3608 | 15,0937 | 18,4207 | 0,8194 | −3,327 | 305,454 |
| 100 000 | 9,6617 | 19,6952 | 23,0259 | 0,8554 | −3,331 | 1 826,379 |
| 1 000 000 | 11,9641 | 24,2999 | 27,6310 | 0,8794 | −3,331 | 11 465,574 |
<!-- /tabelle -->

<!-- tabelle:lorden20 -->
| γ = ARL₀ | h | J(γ) exakt | ln γ / I | Verhältnis J / (ln γ / I) | Differenz J − ln γ / I | Shewhart-Verzögerung |
|---|---|---|---|---|---|---|
| 10 | 0,2909 | 1,3083 | 1,1513 | 1,1364 | 0,157 | 1,309 |
| 30 | 0,9138 | 1,7096 | 1,7006 | 1,0053 | 0,009 | 1,767 |
| 100 | 1,5316 | 2,2672 | 2,3026 | 0,9846 | −0,035 | 2,688 |
| 370 | 2,1754 | 2,9171 | 2,9568 | 0,9866 | −0,040 | 4,605 |
| 1 000 | 2,6651 | 3,4132 | 3,4539 | 0,9882 | −0,041 | 7,257 |
| 10 000 | 3,8137 | 4,5611 | 4,6052 | 0,9904 | −0,044 | 23,361 |
| 100 000 | 4,9650 | 5,7123 | 5,7565 | 0,9923 | −0,044 | 85,036 |
| 1 000 000 | 6,1162 | 6,8635 | 6,9078 | 0,9936 | −0,044 | 339,124 |
<!-- /tabelle -->

**Tabelle F: CUSUM mit zusätzlicher Shewhart-Grenze (Alarm bei S ≥ h oder x > c)**, k = 0,5, ARL₀ = 370 einseitig, h je Grenze c neu abgestimmt (eingefroren, Gitter). Die Grenze c = 3,5 kostet bei δ = 0,5 1,2 % und bei δ = 1 0,5 % und bringt bei δ = 4 das 0,761fache der reinen CUSUM-Verzögerung;
mit c = 3,0 kostet sie 11,8 % bei δ = 0,5 und bringt das 0,677fache bei δ = 4. c = 2,5 ist bei ARL₀ = 370 nicht abstimmbar (die Grenze allein hat ARL₀ = 161). Der Hinweis auf CUSUM mit Shewhart-Grenze steht in Lucas/Saccucci 1990 (Abschnitt 5, gelesen) als Verweis auf Lucas 1982, das nicht gelesen ist; die Zahlen sind eigene Nachrechnung.

<!-- tabelle:hybrid -->
| Zusatzgrenze c | h (ARL₀ = 370) | δ = 0,5 | δ = 1,0 | δ = 2,0 | δ = 3,0 | δ = 4,0 |
|---|---|---|---|---|---|---|
| ∞ (reiner CUSUM) | 4,09548 | 27,675 | 8,573 | 3,406 | 2,231 | 1,744 |
| 4,0 | 4,10023 | 27,703 | 8,576 | 3,385 | 2,125 | 1,520 |
| 3,5 | 4,14688 | 27,993 | 8,620 | 3,335 | 1,969 | 1,328 |
| 3,0 | 4,59761 | 30,927 | 9,156 | 3,304 | 1,776 | 1,181 |
| Shewhart allein | – | 88,890 | 26,746 | 4,605 | 1,705 | 1,126 |
<!-- /tabelle -->

**Das Handbeispiel in Brüchen** (ausgedachtes Zählmodell, Beobachtung x ∈ {0, …, 4}, CUSUM k = 2, h = 3, Shewhart bei x = 4; keine Betriebsdaten): beide Karten haben ARL₀ = 20 (Wahrscheinlichkeiten für x = 0, 1, 2, 3, 4: ohne Änderung 1/5, 1/4, 1/4, 1/4, 1/20; mittlerer Sprung 1/10, 3/20, 1/4, 2/5, 1/10; großer Sprung 0, 1/20, 1/10, 1/5, 13/20; Gegenprobe 1/20, 1/10, 1/5, 2/5, 1/4); die Restlauflängen des CUSUM sind 20, 35/2, 25/2. Bei einem mittleren Sprung (P(x = 4) = 1/10)
braucht der CUSUM 2105/298 ≈ 7,0638, die Shewhart-Karte 10; bei einem großen Sprung (P(x = 4) = 13/20) der CUSUM 2620/1071 ≈ 2,4463, die Shewhart-Karte 20/13 ≈ 1,5385. Der CUSUM gewinnt den kleinen bis mittleren Sprung, die Shewhart-Karte den großen;
Gegenprobe: bei P(x = 4) = 1/4 steht der CUSUM schon bei 1740/427 ≈ 4,0749 gegen 4. Die Reihe x = (3, 3, 3, 0, 4) hat den Pfad S = 1, 2, 3, 1, 3: der CUSUM meldet bei t = 3, die Shewhart-Karte bei t = 5; das Gegenstück x = (0, 4, 3, 1) lässt die Shewhart-Karte bei t = 2 gewinnen.

**Brücke zur Wartezeit.** Die Rekursion S_t = max(0, S_(t−1) + x_t − k) ist dieselbe wie die Lindley-Rekursion der Wartezeit im Ein-Schalter-System (Zuwachs = Bedienzeit minus Zwischenankunftszeit); das ist eine Umbenennung, kein Messbefund.
Gemessen ist: die Rekursion ist gleich dem Maximum der Teilsummen in allen 3905 Reihen der Länge 1 bis 5 mit Werten 0 bis 4, und gleich der Ereignisrechnung der Warteschlange (Ankunfts- und Fertigstellungszeiten) in allen 3125 Reihen der Länge 5.
Das ist Mathematik, keine Aussage darüber, ob Praktiker Wartezeiten so überwachen.

**Rechenzeit (Messung dieser README, Windows, Python 3.14.6, nicht im Test; Zeit nur Hinweis):** exakte Größen von Experiment 1 rund 0,01 s, Experiment 2 0,14 s, Experiment 3 0,59 s, Experiment 4 0,06 s, die Simulation mit 300 Reihen 0,04 s und im schlechtesten Fall (δ = 0,25, ARL₀ = 5000, 470 Reihen) rund 1,2 s (drei Läufe mit den Zufalls-Seeds 35 bis 37: 1,15 bis 1,21 s).

### Zahl → Test

| Zahl oder Aussage der README | Test (Band oder exakt) |
|---|---|
| Kernbotschaft: 8,573 / 26,746, 2,4375, 2,09 / 2,69, 2,03 / 3,80, 2,61, 1,744 / 1,126, 23,71 % / 23,04 %, 2,99fach, h-Werte, 737,12 | `test_claims.py::test_kernbotschaft` (exakt auf die gedruckten Stellen) |
| Tabellen A, B, C, D, E, F (alle Zellen) | `test_claims.py::test_readme_tabellen_gleich_app_tabellen` (exakt, Text gegen Text) und die Orakel-Tests `test_oracle_arl.py`, `test_oracle_umschlag_k.py`, `test_oracle_twosided.py`, `test_oracle_hybrid.py`, `test_frozen_tables.py` (Werte gegen andere Rechenwege) |
| Verteilung: Median, 95-%-Quantil, 0,7386, 23,71 % / 23,04 %, Median 257 / 258 | `test_claims.py::test_verteilung_und_fehlalarm_seite`, `test_oracle_sim.py::test_quantile_und_anteil_der_kette_gegen_handwerte` |
| Simulation: Zufalls-Seeds 35 und 2026, Abdeckung in 40 Zufalls-Seeds | `test_claims.py::test_simulation_zufalls_seeds_35_und_2026` (Band), `test_claims.py::test_abdeckung_ueber_40_zufalls_seeds` (Anzahl) |
| k\*/δ = 0,5, ARL₁\*, 2,991, 2,043, 1,427, 3,756 | `test_claims.py::test_bestes_k_und_der_preis_eines_falschen_k`, `test_oracle_umschlag_k.py::test_bestes_k_ist_delta_halbe_gegen_rastersuche` |
| Lorden-Näherung: Vorzeichen, 0,0064, 11 465,574, 471,8 | `test_claims.py::test_lorden_naeherung`, `test_oracle_umschlag_k.py` |
| Umschlag zweiseitig 2,6125 | `test_claims.py::test_zweiseitig_umschlag_und_zaehlung`, `test_frozen_tables.py` |
| quasistationär 7,9038 gegen 8,5730 | `test_eigenschaften_cusum.py::test_quasistationaere_verzoegerung_ist_kleiner_als_zero_state_und_nicht_das_shiryaev_mass` |
| Hälftenregel 369,9898 / 369,9873, 10⁻¹¹, 2·10⁻¹⁵ | `test_oracle_twosided.py::test_zweiseitig_bei_delta_null_ist_die_haelfte_der_einseitigen_arl`, `…::test_haelftenregel_auf_einem_beobachtungsgitter_gilt_auf_rundungsfehler` |
| gemischte Quellentabelle 74,2241 / 77,0785, 37,9962 / 38,0096 | `test_claims.py::test_gemischte_quellentabelle` |
| Hybrid: Zahlen, 1,2 % / 0,5 % / 11,8 % / 0,761fache / 0,677fache, ARL₀ 370,01, c = 2,5 nicht abstimmbar (161) | `test_claims.py::test_hybrid_zahlen_der_readme`, `test_frozen_tables.py`, `test_oracle_hybrid.py::test_zu_kleine_grenze_ist_nicht_abstimmbar` |
| Handbeispiel (Brüche, Pfade, 3905 und 3125 Reihen) | `test_handbeispiel.py`, `test_claims.py::test_handbeispiel_zahlen_der_readme` |
| Hilfetexte der Presets (3,2fache, 1,55fache, 2,99fache, 2,09 / 2,44, 0,226 / 0,321, 1,1513 / 1,3083) | `test_claims.py::test_zahlen_der_preset_hilfetexte` |
| Rechenbudget (470 Reihen, 1 000 000 Normalzahlen) | `test_claims.py::test_rechenbudget` |
| Rechenzeiten, 1,2 µs je Zahl, 200 s der Tabellen | Messung dieser README, nicht im Test |
| Annahmentabelle in App und README gleich | `test_claims.py::test_annahmentabelle_in_app_und_readme_gleich` |

## Ehrliche Grenzen

* **Bekanntes Ziel und bekannte Streuung, unabhängig, normal.** Geschätzte Parameter, Autokorrelation und Schiefe sind hier nicht gemessen; wie weit die k-Tabelle dort gilt, ist offen. Die Quellenlage sagt nur, dass die Annahme einer bekannten Nach-Änderungs-Größe schwach ist (Laub u. a., arXiv 2006.04576; Yamin u. a., arXiv 2211.12091: die Größe wurde dort nach Ansicht der Daten gewählt bzw. als bekannt genommen).
* **Der CUSUM ist nicht robust gegen ein falsches k** (bis zum 2,991fachen der besten Verzögerung auf dem gemessenen Raster); „CUSUM ist optimal“ gilt für bekanntes δ, im Lorden-Maß.
* **Ein Urteil über „der CUSUM ist besser“ braucht Voraussetzungen:** bei gleicher ARL₀ = 370, k = δ/2, in den acht geprüften δ (bei δ = 4 knapp: 1,121 gegen 1,126). Mit k = 0,5 verliert er ab δ = 2,4375.
* **Eine Einzelreihe ist ein Fall, keine Regel:** im Handbeispiel lässt x = (0, 4, 3, 1) die Shewhart-Karte gewinnen.
* **Fehlalarm-Verteilung „fast gleich“** heißt 23,71 % gegen 23,04 % vor Beobachtung 100 und Median 257 gegen 258, nicht Gleichheit.
* **Kein belegter Nutzen:** ob eine Alarmregel in Logistik, Lager oder Fertigung bessere Entscheidungen liefert, ist in keiner gelesenen Quelle gemessen; die Reihen sind ausgedacht.
* **Plattform:** Zahlen unter Windows, Python 3.14.6; Brüche und Ganzzahlen plattformfest, Gleitkomma-ARL mit relativer Toleranz, Simulationsgrößen als Band.

Die Annahmen als Tabelle (dieselbe Fassung wie in der App, „🚧 Wo die Annahmen enden“):

<!-- annahmen -->
| Annahme | Was hier gilt, und was nicht gemessen ist | Wer setzt an |
|---|---|---|
| **Ziel 0 und σ = 1 sind bekannt** (die Beobachtung ist standardisiert) | Alle Zahlen setzen das voraus. Geschätzte Parameter sind nicht gemessen. | Stück „Parameter schätzen“ der Linie |
| **Unabhängige, normalverteilte Beobachtungen** | Autokorrelation und Schiefe sind nicht gemessen; die k-Tabelle gilt nur unter dieser Annahme. | Stück „Abhängige Werte“ der Linie |
| **Der Sprung ist konstant, einseitig und liegt zu Beginn (ν = 0) oder bei ν** | Einseitig gerechnet; zweiseitig nur für k ∈ {0,25; 0,5; 0,75; 1} und ARL₀ ∈ {100; 370; 1000} (eingefroren). | – |
| **Das Ziel ARL₀ = 370 ist gesetzt** (die 3σ-Karte zweiseitig hat 370,398) | Schwellen h gehören zu einem Ziel: 4,0954 einseitig und 4,7738 zweiseitig gelten für 370, 4,0965 und 4,7749 für 370,398. | – |
| **Maß: zero-state, hier gleich dem Lorden-Maß** | Quasistationär, zyklisch (Shiryaev) und Anfangsvorsprung sind nicht Stoff dieser Seite. | Stück „EWMA und das Maß“ der Linie |
| **Kein Betriebsbeleg** | Ob eine Alarmregel in Logistik, Lager oder Fertigung bessere Entscheidungen liefert, ist hier nicht gemessen. | – |
<!-- /annahmen -->

## 🚧 Bewusst nicht enthalten, nicht gemessen

* **Zweiseitig für andere k und ARL₀** als k ∈ {0,25; 0,5; 0,75; 1}, ARL₀ ∈ {100; 370; 1000}: nicht gemessen (App meldet statt einer Zahl). Zweiseitig gibt es keine Hüllkurve k = δ/2.
* **Steady-state und Pollak-Maß des CUSUM mit Shewhart-Grenze**, Shiryaev–Roberts, Anfangsvorsprung: nicht gemessen und nicht Stoff dieser Seite (Stück „EWMA und das Maß“). Quasistationär ist hier nur der Kontrollwert des reinen CUSUM (7,9038).
* **Nicht normal, abhängig, geschätzt:** nicht Stoff dieser Seite (Stücke „Parameter schätzen“ und „Abhängige Werte“).
* **EWMA, Zusatzregeln der Karte, Western-Electric-Regeln:** nicht Stoff dieser Seite.
* **Regression, Prognose-Zurücksetzen, Bestätigungsregel, Kosten** (dort `niveausprung-demo`): hier nicht.
* **Kein Literaturtest** gegen Page 1954, Lorden 1971 (nur Abstract), Moustakides 1986, Lucas 1982, Brook/Evans 1972 (nicht gelesen), gegen NIST-Tabellenzahlen und die SADD-Spalten von Moustakides u. a. 2009; die ARL₀-Spalten von Moustakides u. a. 2009 sind **nicht** getestet (ihre Zahlen liegen dem Bau nicht im Wortlaut vor).
* **PDF-Export**, Download der Tabellen: nicht umgesetzt.
* **Linux-Lauf der Tests (Docker-Prüfung) und Streamlit-Cloud-Rechenzeiten:** nicht gemessen; Zahlen nur unter Windows.
* **Mausklick im Browser:** AppTest klickt Presets und Regler, der Browser wurde nur über die HTTP-Gesundheitsprüfung des gestarteten Servers geprüft (siehe Verifikation).

## Quellen (Stufe: W = gelesen, sonst nicht gelesen)

* Moustakides, Polunchenko, Tartakovsky 2009, arXiv 0908.4119 – **W** (CUSUM-Definition, Integralgleichung, „J_L = SADD = E₀[T]“; ihre SADD-Spalten sind nicht zitiert).
* Moustakides 2008, arXiv 0804.0741 – **W** (Einleitung, Abschnitt 3). Bartroff, Tartakovsky 2024, arXiv 2403.18782 – **W** (Satz 6: CUSUM als Minimum wiederholter einseitiger SPRT-Zeiten).
* Lucas, Saccucci 1990, Technometrics 32 – **W** (Anhang A: Brook/Evans-Verfahren; Abschnitt 5: CUSUM mit Shewhart-Grenze als Verweis; Tab. 6, Spaltenzuordnung nur durch Rechnung bestätigt).
* Baron, Malov, arXiv 2502.09185 – **W** (Abstract, Abschnitt 1: die CUSUM-Rekursion ist die Lindley-Rekursion). Zwetsloot, Jones-Farmer, Woodall 2024, Quality Engineering 36 – **W**; Zwetsloot, Woodall 2017 – **W** (Zusammenfassung, Tabellenzeile H = 4,17). Laub u. a., arXiv 2006.04576, und Yamin u. a., arXiv 2211.12091 – **W**.
* NIST/SEMATECH 6.3.2.3 – **W über die Zusammenfassung des Abrufwerkzeugs** („k = ½ der Verschiebung“; keine Tabellenzahl zitiert).
* **Nicht gelesen:** Page 1954, Brook/Evans 1972, Lorden 1971 (nur Abstract), Moustakides 1986, Lucas 1982, Montgomery, Hawkins/Olwell, Basseville/Nikiforov, Siegmund. **Keine Neuheit:** alles ist Lehrbuchstoff; neu ist die Vorführbarkeit mit Zähleinheit, zwei exakten Wegen und der k-Abhängigkeit. Befunde ohne Quelle sind eigene Nachrechnung in diesem Modell.

## Verifikation

* **Handwerte in Brüchen** (`test_handbeispiel.py`): Übergangsmatrix, ARL₀ = 20 für beide Karten, Lauflängenverteilung (Matrixweg gegen Aufzählung aller Folgen), ARL₁-Brüche, Pfade, Maximum der Teilsummen in allen 3905 Reihen, Ereignisrechnung der Warteschlange in allen 3125 Reihen der Länge 5.
* **Orakel** (`test_oracle_*.py`, `test_literatur_cusum.py`): Integralgleichung gegen Gitter (Richardson), Gitterzellen gegen Quadratur der Normaldichte, Simulation (SplitMix64, Band von 4 Standardfehlern) gegen die exakten Größen in 8 Zellen, 2D-Kette gegen Hälftenregel, Beobachtungsgitter, Simulation und die Literaturwerte (Lucas/Saccucci Tab. 6, Zwetsloot/Woodall H = 4,17), Umschlag und bestes k gegen Raster, Hybrid-Gitter gegen Simulation.
* **Eigenschaften, Randfälle, Claims, Verdikte, Presets, Layout, App** (AppTest über jeden Button, jeden Regler an Minimum und Maximum, Eckfälle), Footer, Laufzeit-Importe, `modul_frisch`, eingefrorene Tabellen gegen Neurechnung.
* **241 Tests**, Laufzeit der Suite rund 210 s (Windows, Python 3.14.6, ein Prozess).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kopf, Expander, Ergebnis, Kernfragen, vier Experimente, Tabelle der Zusatzgrenze, Annahmen, Formeln |
| `cus_arl.py` | exakte Rechnung: Integralgleichung, Gitter, 2D-Kette, Hybrid, Abstimmung, Umschlag, bestes k, Lauflängenverteilung |
| `cus_chart.py` | Karten als Rekursionen, Läufe bis zum Alarm, Simulation |
| `cus_rng.py` | SplitMix64 (wörtliche Kopie der Vorlage des Website-Repos) |
| `cus_scenario.py` | die Reihe mit Sprung |
| `cus_handbeispiel.py` | das Handbeispiel in Brüchen |
| `cus_reference.py` | lädt die eingefrorenen Tabellen |
| `cus_evaluation.py` | Experimente, Kennzahlen, Verdikte als reine Funktionen |
| `cus_constants.py`, `cus_presets.py`, `cus_layout.py`, `cus_visualization.py`, `cus_texts.py` | Regler und Presets, Permalink, Kopf, Abbildungen, Texte und Tabellen |
| `cus_twosided.json`, `cus_hybrid.json` | eingefrorene Tabellen (Quelle, Stand, Verfahren im Feld `_meta`) |
| `tools/cus_make_tables.py` | erzeugt und prüft die eingefrorenen Tabellen |
| `modul_frisch.py` | lädt geänderte Hilfsmodule nach einem Update neu (Streamlit Cloud) |
| `requirements-dev.txt`, `requirements.txt` | Laufzeit plus pytest; Laufzeit (ungepinnt) |
| `tests/` | Handwerte, Orakel, Literatur, Eigenschaften, Randfälle, Claims, Verdikte, Presets, Layout, App, Footer, Laufzeit-Importe, eingefrorene Tabellen |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die CI läuft mit Python 3.12, lokal lief alles mit Python 3.14.6; die in der Streamlit Cloud eingestellte Version ist nicht geprüft. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python tools/cus_make_tables.py --check       # drei Zellen je Datei neu rechnen; --lang: alle Zellen
```

Gebaut mit Streamlit, Plotly, NumPy und SciPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
