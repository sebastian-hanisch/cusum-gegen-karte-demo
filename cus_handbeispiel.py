"""Das Handbeispiel in Brüchen (kleines, ausgedachtes Zählmodell, kein Betriebsmodell): CUSUM mit drei Zuständen gegen Shewhart-Karte, beide ARL0 = 20.

Beobachtung x in {0, 1, 2, 3, 4}. Ohne Änderung P0 = (4, 5, 5, 5, 1) / 20 (P(x = 4) = 1/20). Shewhart: Alarm bei x = 4, ARL0 = 1/p = 20.
CUSUM mit k = 2 und h = 3: Zustände S in {0, 1, 2}; Zuwachs y = x - 2 in {-2, -1, 0, +1, +2}. Alles in `fractions.Fraction`, keine Rundung.
Die Gleichheit der beiden ARL0 ist eine Wahl des Beispiels (durch Suche gefunden), kein Phänomen."""

from fractions import Fraction as F

WERTE = (0, 1, 2, 3, 4)
K, H, C = 2, 3, 4
P0 = (F(1, 5), F(1, 4), F(1, 4), F(1, 4), F(1, 20))
P_MITTEL = (F(1, 10), F(3, 20), F(1, 4), F(2, 5), F(1, 10))   # P(x = 4) verdoppelt, Mittel 2,25
P_GROSS = (F(0), F(1, 20), F(1, 10), F(1, 5), F(13, 20))      # P(x = 4) = 13/20, Mittel 3,45


def transition_matrix(p, k: int = K, h: int = H):
    """Übergangsmatrix der alarmfreien Zustände 0..h-1; der Rest jeder Zeile ist die Alarmwahrscheinlichkeit."""
    r = [[F(0)] * h for _ in range(h)]
    for s in range(h):
        for q, x in zip(p, WERTE):
            t = max(0, s + x - k)
            if t < h:
                r[s][t] += q
    return r


def solve_linear(a, b):
    """Gauß-Elimination in Brüchen: löst a x = b."""
    n = len(b)
    m = [[F(v) for v in row] + [F(b[i])] for i, row in enumerate(a)]
    for col in range(n):
        piv = next(r for r in range(col, n) if m[r][col] != 0)
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        m[col] = [v / pv for v in m[col]]
        for r in range(n):
            if r != col and m[r][col] != 0:
                f = m[r][col]
                m[r] = [x - f * y for x, y in zip(m[r], m[col])]
    return [m[i][n] for i in range(n)]


def cusum_run_lengths(p, k: int = K, h: int = H):
    """L(s) = mittlere Zahl der Beobachtungen bis zum Alarm ab Zustand s, als Brüche (lineares System (I - R) L = 1)."""
    r = transition_matrix(p, k, h)
    n = len(r)
    return solve_linear([[(1 if i == j else 0) - r[i][j] for j in range(n)] for i in range(n)], [1] * n)


def arl_cusum(p, k: int = K, h: int = H):
    return cusum_run_lengths(p, k, h)[0]


def arl_shewhart(p, c: int = C):
    return 1 / sum(q for q, x in zip(p, WERTE) if x >= c)


def cusum_pmf(p, tmax: int, k: int = K, h: int = H):
    """P(Lauflänge = t), t = 1..tmax, als Brüche (Matrixweg e_0 R^(t-1) a)."""
    r = transition_matrix(p, k, h)
    alarm = [1 - sum(row) for row in r]
    v = [F(1)] + [F(0)] * (h - 1)
    out = []
    for _ in range(tmax):
        out.append(sum(vi * ai for vi, ai in zip(v, alarm)))
        v = [sum(v[i] * r[i][j] for i in range(h)) for j in range(h)]
    return out


def shewhart_pmf(p, tmax: int, c: int = C):
    q = sum(x_p for x_p, x in zip(p, WERTE) if x >= c)
    return [(1 - q) ** (t - 1) * q for t in range(1, tmax + 1)]


def cusum_pfad(xs, k: int = K):
    """Pfad S_1..S_n (läuft über den Alarm hinaus weiter)."""
    s = 0
    out = []
    for x in xs:
        s = max(0, s + x - k)
        out.append(s)
    return out


def alarm_times(xs, k: int = K, h: int = H, c: int = C):
    """(Alarmzeit Shewhart, Alarmzeit CUSUM) einer Reihe, 1-basiert, None ohne Alarm."""
    ts = next((t for t, x in enumerate(xs, start=1) if x >= c), None)
    tc = next((t for t, s in enumerate(cusum_pfad(xs, k), start=1) if s >= h), None)
    return ts, tc
