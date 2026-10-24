"""
Características para los agentes que aprenden.

Cada decisión se describe con un vector global del estado (lo que ve el que
juega) y, para cada acción legal, un vector de la acción: qué especie juega,
cuántas cartas se lleva, cómo queda su mano para formar bandadas, etc. Los
agentes puntúan cada acción con [global, acción] y eligen entre las legales,
así una misma red sirve para jugar, robar y formar bandadas en mesas de 2 a 5.

La misma codificación está en web/engine.js (features) para que los modelos
entrenados en Python jueguen en el navegador; tests/test_features.py lo
comprueba.
"""

import numpy as np

from cubirds.game.engine import (
    NS, SMALL, BIG, PLAY, DRAW, A_DRAW, A_NODRAW, A_FLOCK0, A_NOFLOCK, Game, decode_play,
)
from cubirds.game.bots import progress, hand_value, AFINADO, DEFAULT

G_DIM = 33
A_DIM = 32
DIM = G_DIM + A_DIM


def global_features(g, p=None):
    p = g.cur if p is None else p
    h, col = g.hands[p], g.cols[p]
    opp = [q for q in range(g.n) if q != p]
    x = np.zeros(G_DIM, dtype=np.float32)
    for s in range(NS):
        x[s] = h[s] / BIG[s]
        x[8 + s] = min(col[s], 3) / 3
        x[17 + s] = max(min(g.cols[q][s], 3) for q in opp) / 3
    x[16] = progress(col) / 4
    x[25] = min(progress(g.cols[q]) for q in opp) / 4
    x[26] = sum(sum(g.hands[q]) for q in opp) / len(opp) / 10
    x[27] = sum(h) / 10
    x[28] = len(g.deck) / 60
    x[29] = (g.n - 2) / 3
    x[30 + (0 if g.phase == PLAY else 1 if g.phase == DRAW else 2)] = 1.0
    return x


def action_features(g, a, p=None):
    p = g.cur if p is None else p
    h, col = g.hands[p], g.cols[p]
    x = np.zeros(A_DIM, dtype=np.float32)
    if a < 64:
        s, r, side = decode_play(a)
        row = g.rows[r]
        k = Game.enclosed(row, s, side)
        taken = row[:k] if side == 0 else row[len(row) - k:]
        nh = list(h)
        nh[s] = 0
        for c in taken:
            nh[c] += 1
        x[s] = 1.0
        x[8] = 1.0
        x[9] = h[s] / BIG[s]
        x[10] = k / 6
        for c in taken:
            x[11 + c] += 1.0 / BIG[c]
        x[19] = sum(1 for t in range(NS) if nh[t] >= SMALL[t]) / 3
        x[20] = sum(1 for t in range(NS) if nh[t] >= BIG[t]) / 2
        x[21] = (hand_value(nh, col, DEFAULT) - hand_value(h, col, DEFAULT)) / 3
        x[22] = (hand_value(nh, col, AFINADO) - hand_value(h, col, AFINADO)) / 5
        x[23] = 1.0 if k == 0 else 0.0
        x[24] = (len(row) - k + h[s]) / 10
    elif a == A_DRAW:
        x[25] = 1.0
    elif a == A_NODRAW:
        x[26] = 1.0
    elif a == A_NOFLOCK:
        x[30] = 1.0
    else:
        s = a - A_FLOCK0
        keep = 2 if h[s] >= BIG[s] else 1
        nc = list(col)
        nc[s] += keep
        x[s] = 1.0
        x[27] = 1.0
        x[28] = keep / 2
        x[29] = (progress(col) - progress(nc)) / 2
        x[31] = min(nc[s], 3) / 3
    return x


def decision(g):
    """(acciones legales, matriz [n_legal, DIM]) para el jugador de turno."""
    legal = g.legal_actions()
    gx = global_features(g)
    X = np.empty((len(legal), DIM), dtype=np.float32)
    X[:, :G_DIM] = gx
    for i, a in enumerate(legal):
        X[i, G_DIM:] = action_features(g, a)
    return legal, X
