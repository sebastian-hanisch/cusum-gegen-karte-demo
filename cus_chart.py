"""Kontrollkarten als Rekursionen und Läufe bis zum Alarm (reiner Python-Kern, Zufall nur aus cus_rng).

Shewhart-Karte (einseitig): Alarm bei x_t > c. CUSUM (einseitig, nach oben): S_0 = 0, S_t = max(0, S_(t-1) + x_t - k), Alarm bei S_t >= h.
Zweiseitig: zwei solche Statistiken (S+ mit x, S- mit -x), Alarm bei einer von beiden; Shewhart zweiseitig |x| > c.
Zähleinheit: Beobachtungen; t = 1, 2, ...; die Alarmbeobachtung zählt mit."""

from statistics import NormalDist

from cus_rng import SplitMix64

_INV = NormalDist().inv_cdf
_U_MIN = 2.0 ** -53


def normal_from(rng: SplitMix64) -> float:
    """Standardnormalzahl aus einer SplitMix64-Gleichverteilten (Inversionsverfahren; u = 0 wird auf 2^-53 gesetzt)."""
    return _INV(max(rng.random(), _U_MIN))


def cusum_path(xs, k: float):
    """Pfad S_1..S_n des einseitigen CUSUM (läuft über den Alarm hinaus weiter)."""
    s = 0.0
    out = []
    for x in xs:
        s = max(0.0, s + x - k)
        out.append(s)
    return out


def cusum_path_two(xs, k: float):
    """Pfade (S+_t, S-_t) des zweiseitigen CUSUM."""
    sp = sm = 0.0
    up, dn = [], []
    for x in xs:
        sp = max(0.0, sp + x - k)
        sm = max(0.0, sm - x - k)
        up.append(sp)
        dn.append(sm)
    return up, dn


def first_index(values, predicate):
    """Erster 1-basierter Index, an dem predicate(value) gilt, sonst None."""
    for t, v in enumerate(values, start=1):
        if predicate(v):
            return t
    return None


def shewhart_alarm(xs, c: float, two: bool = False):
    return first_index(xs, (lambda x: abs(x) > c) if two else (lambda x: x > c))


def cusum_alarm(xs, k: float, h: float):
    return first_index(cusum_path(xs, k), lambda s: s >= h)


def draw_series(seed: int, n: int, delta: float, nu: int):
    """Eine Reihe der Länge n: x_t ~ N(0,1) für t <= nu, N(delta, 1) danach; gleicher Zufalls-Seed gibt gleiche Reihe."""
    rng = SplitMix64(seed)
    return [normal_from(rng) + (delta if t > nu else 0.0) for t in range(1, n + 1)]


def simulate_delays(seed: int, runs: int, delta: float, c: float, k: float, h: float, tcap: int):
    """Läuft `runs` Reihen mit dem Sprung vor der ersten Beobachtung (nu = 0) bis beide Karten gemeldet haben.

    Gibt (Lauflängen Shewhart, Lauflängen CUSUM, Zahl der Normalzahlen) zurück; je Reihe laufen beide Karten auf denselben Zahlen.
    Reihen ohne Alarm bis tcap zählen mit tcap (Notbremse, bei delta >= 0,25 und ARL0 <= 5000 praktisch nie)."""
    rng = SplitMix64(seed)
    ts_all, tc_all = [], []
    used = 0
    for _ in range(runs):
        s = 0.0
        ts = tc = 0
        t = 0
        while not (ts and tc) and t < tcap:
            t += 1
            x = _INV(max(rng.random(), _U_MIN)) + delta
            if not ts and x > c:
                ts = t
            if not tc:
                s = s + x - k
                if s < 0.0:
                    s = 0.0
                if s >= h:
                    tc = t
        used += t
        ts_all.append(ts or tcap)
        tc_all.append(tc or tcap)
    return ts_all, tc_all, used


def simulate_false_alarms(seed: int, runs: int, horizon: int, c: float, k: float, h: float):
    """Zahl der Reihen ohne Sprung (delta = 0), in denen die Shewhart-Karte bzw. der CUSUM innerhalb der ersten `horizon` Beobachtungen meldet.

    Gibt (Anzahl Shewhart, Anzahl CUSUM, Normalzahlen) zurück; beide Karten laufen auf denselben Zahlen."""
    rng = SplitMix64(seed)
    n_s = n_c = 0
    used = 0
    for _ in range(runs):
        s = 0.0
        ts = tc = False
        for _t in range(horizon):
            x = _INV(max(rng.random(), _U_MIN))
            used += 1
            if x > c:
                ts = True
            s = s + x - k
            if s < 0.0:
                s = 0.0
            if s >= h:
                tc = True
            if ts and tc:
                break
        n_s += ts
        n_c += tc
    return n_s, n_c, used


def mean_se(values):
    """Mittel und Standardfehler des Mittels (n - 1 im Nenner der Varianz)."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, (var / n) ** 0.5
