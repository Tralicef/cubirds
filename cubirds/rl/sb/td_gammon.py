"""
Estilo TD-Gammon (Sutton & Barto, sección 16.1): red neuronal que valúa
afterstates, entrenada con TD(λ) y trazas de elegibilidad sobre sus pesos, y
self-play (los rivales son copias recientes de la misma red, mezcladas con
bots para no quedar encerrada en su propio estilo).
"""

import numpy as np

from cubirds.game.bots import Afinado
from cubirds.game.engine import PLAY
from cubirds.rl.features import DIM, decision
from cubirds.rl.sb.common import OPPONENTS, TABLE_OPP, Episode

HIDDEN = 64


def _v(params, X):
    W1, b1, W2, b2 = params
    return np.maximum(X @ W1.T + b1, 0) @ W2 + b2


class SnapshotAgent(Afinado):
    """Rival de self-play: juega con una copia de la red; roba y forma bandadas como Afinado."""

    def __init__(self, params):
        self.params = params

    def play(self, g, legal, rng):
        return legal[int(np.argmax(_v(self.params, decision(g)[1])))]


class TDGammon:
    chapter = "16.1"
    label = "TD-Gammon (red + TD(λ) + self-play)"
    desc = "Red que valúa afterstates, entrenada con TD(λ) y trazas sobre sus pesos, jugando contra copias de sí misma."

    def __init__(self, rng, **hp):
        self.rng = rng
        self.hp = hp
        nr = np.random.default_rng(rng.randrange(2**31))
        self.W1 = nr.normal(0, 1 / np.sqrt(DIM), (HIDDEN, DIM))
        self.b1 = np.zeros(HIDDEN)
        self.W2 = nr.normal(0, 1 / np.sqrt(HIDDEN), HIDDEN) * 0.1
        self.b2 = 0.0
        self.snapshot = self._copy()
        self.t = 0
        self.episodes = 0

    def _copy(self):
        return (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2)

    def _params(self):
        return (self.W1, self.b1, self.W2, self.b2)

    def make_episode(self, rng):
        names, weights = zip(*OPPONENTS)
        k = rng.choice(TABLE_OPP)
        opps = [SnapshotAgent(self.snapshot) if rng.random() < self.hp.get("self_play", 0.5)
                else rng.choices(names, weights)[0] for _ in range(k)]
        return Episode(rng, opponents=opps)

    # --- política ---
    def greedy(self, X):
        return int(np.argmax(_v(self._params(), X)))

    def choose(self, ep):
        return self.greedy(decision(ep.g)[1].astype(np.float64))

    def play(self, ep):
        alpha, lam, eps = self.hp.get("alpha", 0.003), self.hp.get("lam", 0.7), self.hp.get("epsilon", 0.05)
        z = [np.zeros_like(self.W1), np.zeros_like(self.b1), np.zeros_like(self.W2), 0.0]
        prev_v = prev_r = None
        while not ep.done:
            X = decision(ep.g)[1].astype(np.float64)
            j = self.rng.randrange(len(X)) if self.rng.random() < eps else self.greedy(X)
            x = X[j]
            h_pre = self.W1 @ x + self.b1
            h = np.maximum(h_pre, 0)
            v = h @ self.W2 + self.b2
            mask = (h_pre > 0).astype(float)
            grad = [np.outer(self.W2 * mask, x), self.W2 * mask, h, 1.0]
            r = ep.step(j)
            if prev_v is not None:
                self._td(z, prev_v, prev_r + v, alpha)
            z = [lam * zi + gi for zi, gi in zip(z, grad)]
            prev_v, prev_r = v, r
            self.t += 1
        if prev_v is not None:
            self._td(z, prev_v, prev_r, alpha)
        self.episodes += 1
        if self.episodes % self.hp.get("snapshot_every", 500) == 0:
            self.snapshot = self._copy()

    def _td(self, z, v, target, alpha):
        delta = target - v
        self.W1 += alpha * delta * z[0]
        self.b1 += alpha * delta * z[1]
        self.W2 += alpha * delta * z[2]
        self.b2 += alpha * delta * z[3]

    def export_data(self):
        r = lambda a: np.round(a, 5).tolist()
        return {"kind": "valor", "W1": r(self.W1.T), "b1": r(self.b1), "W2": r(self.W2), "b2": round(float(self.b2), 5)}


ALGORITHMS = {"td_gammon": (TDGammon, dict(alpha=0.003, lam=0.7, epsilon=0.05, self_play=0.5, snapshot_every=500))}
