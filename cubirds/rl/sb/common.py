"""
Base común para los algoritmos de Sutton & Barto (Reinforcement Learning: An
Introduction, 2ª ed.) aplicados a CuBirds.

Idea central: valores de *afterstate* (sección 6.8). En vez de valuar el
estado completo (gigante), cada jugada posible se describe por la situación en
que quedaría (después de jugar y llevarse los pájaros encerrados) y los
algoritmos aprenden el valor de esa situación. Elegir jugada = elegir el
afterstate de mayor valor esperado. Robar y formar bandadas se deciden con
las reglas de Afinado: lo que se aprende es la jugada principal.

- Métodos tabulares (Parte I): el afterstate se discretiza en una clave chica
  (ver `key`), ~32.000 combinaciones posibles (unas pocas miles aparecen).
- Métodos aproximados (Parte II): se usan las características por jugada de
  cubirds/rl/features.py.

Recompensa: 1 al ganar (repartido si hay empate) y, con shaping, entre una
decisión y la siguiente el cambio del potencial Φ = 0,1·pájaros − 0,1·bandadas
que faltan (Ng et al., con Φ = 0 al terminar, que no cambia la política óptima).

Ojo con el shaping en los métodos tabulares que miran hasta el final: el
retorno desde una jugada queda R − Φ(estado antes de jugar), y ese Φ no depende
de la jugada pero sí del estado, que la clave no ve entera; mezclado en el
valor de cada clave la confunde (por ejemplo, "no se lleva nada" parece mejor
porque es más común al principio, cuando Φ es bajo). Sin shaping es peor: la
probabilidad de ganar cambia apenas unos puntos entre una jugada y otra del
mismo turno y, con 10.000 partidas, el ruido de cada clave es mayor que esa
diferencia. Los métodos aproximados no sufren el sesgo, porque sus
características incluyen el estado y pueden descontar ese término.

La misma clave está en web/engine.js (sbKey).
"""

import random

from cubirds.game.bots import AFINADO, Afinado, hand_value, make_agent, progress
from cubirds.game.engine import NS, SMALL, BIG, PLAY, Game, decode_play

OPPONENTS = [("afinado", 0.35), ("coleccionista", 0.25), ("goloso", 0.25), ("azar", 0.15)]
TABLE_OPP = [1, 2, 2, 3, 3, 4]
EVAL_TABLES = [["afinado"], ["afinado", "coleccionista"], ["afinado", "goloso", "coleccionista"],
               ["afinado", "afinado", "coleccionista", "goloso"]]


def potential(g, p):
    return 0.1 * sum(g.cols[p]) - 0.1 * progress(g.cols[p])


def _bucket(x, edges):
    i = 0
    while i < len(edges) and x >= edges[i]:
        i += 1
    return i


def key(g, a, p=None):
    """Clave discreta del afterstate de la jugada a (0..63) para el jugador p."""
    p = g.cur if p is None else p
    h, col = g.hands[p], g.cols[p]
    s, r, side = decode_play(a)
    row = g.rows[r]
    k = Game.enclosed(row, s, side)
    taken = row[:k] if side == 0 else row[len(row) - k:]
    nh = list(h)
    nh[s] = 0
    for c in taken:
        nh[c] += 1
    ready, gain = 0, 0
    before = progress(col)
    for t in range(NS):
        if nh[t] >= SMALL[t]:
            keep = 2 if nh[t] >= BIG[t] else 1
            ready = max(ready, keep)
            nc = list(col)
            nc[t] += keep
            gain = max(gain, before - progress(nc))
    opp = min(progress(g.cols[q]) for q in range(g.n) if q != p)
    return (
        _bucket(k, [1, 2, 3, 5]),                 # pájaros que se lleva: 0, 1, 2, 3-4, 5+
        ready,                                    # mejor bandada posible: 0 ninguna, 1 chica, 2 grande
        min(gain, 2),                             # cuánto acerca a ganar esa bandada
        _bucket(before, [2, 3, 4]),               # bandadas que le faltan: 0-1, 2, 3, 4+
        _bucket(opp, [2, 3]),                     # al mejor rival: 0-1, 2, 3+
        _bucket(h[s] / SMALL[s], [0.5, 1.0]),     # cartas que quema respecto de una bandada chica
        _bucket(sum(nh), [1, 4, 8]),              # mano después: 0 (ronda nueva), 1-3, 4-7, 8+
        # cuánto mejora la mano según la evaluación de Afinado
        _bucket(hand_value(nh, col, AFINADO) - hand_value(h, col, AFINADO), [-0.5, 0.5, 1.5, 3.0]),
    )


def key_str(k):
    return ",".join(map(str, k))


class Episode:
    """Una partida desde un asiento: el agente decide sólo sus jugadas.

    keys() da las claves de las jugadas legales; step(i) juega la i-ésima, deja
    que pasen robar/bandada (reglas de Afinado) y los turnos de los rivales, y
    devuelve la recompensa hasta la próxima decisión (o el final)."""

    def __init__(self, rng, opponents=None, shaping=True):
        self.shaping = shaping
        if opponents is None:
            names, w = zip(*OPPONENTS)
            opponents = rng.choices(names, w, k=rng.choice(TABLE_OPP))
        self.n = len(opponents) + 1
        self.me = rng.randrange(self.n)
        self.g = Game(self.n, rng.randrange(1 << 30))
        it = iter(opponents)
        # cada rival es el nombre de un bot o un agente ya armado (self-play)
        self.agents = [None if q == self.me else (lambda o: make_agent(o) if isinstance(o, str) else o)(next(it))
                       for q in range(self.n)]
        self.rules = Afinado()
        self._advance()
        self.phi = potential(self.g, self.me)

    def _advance(self):
        g = self.g
        while not g.done and not (g.cur == self.me and g.phase == PLAY):
            agent = self.rules if g.cur == self.me else self.agents[g.cur]
            g.step(agent.act(g, g.rng))

    @property
    def done(self):
        return self.g.done

    def legal(self):
        return self.g.legal_actions()

    def keys(self):
        return [key(self.g, a) for a in self.g.legal_actions()]

    def step(self, i):
        g = self.g
        g.step(g.legal_actions()[i])
        self._advance()
        if g.done:
            win = (1.0 / len(g.winners)) if self.me in g.winners else 0.0
            return win - self.phi if self.shaping else win
        if not self.shaping:
            return 0.0
        phi = potential(g, self.me)
        r = phi - self.phi
        self.phi = phi
        return r


def sample_episode(rng, shaping=True):
    return Episode(rng, shaping=shaping)


def evaluate(choose, games=300, seed=12345):
    """
    Índice de victorias en mesas fijas (1.0 = lo justo, ganar 1 de cada n).
    choose(ep) -> índice de la jugada legal.
    """
    out = {}
    for tab in EVAL_TABLES:
        wins = 0.0
        for k in range(games):
            ep = Episode(random.Random(seed + k), opponents=tab)
            while not ep.done:
                ep.step(choose(ep))
            g = ep.g
            wins += (1.0 / len(g.winners)) if ep.me in g.winners else 0.0
        out[" + ".join(tab)] = round(wins / games * (len(tab) + 1), 3)
    return out


def summary(ev):
    return sum(ev.values()) / len(ev)
