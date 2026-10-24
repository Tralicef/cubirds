"""
Bots heurísticos de CuBirds. La misma lógica está en web/engine.js (objeto H).

Todos miran sólo lo que ve un jugador: su mano, las filas, las colecciones
y los tamaños de las manos ajenas.
"""

from .engine import (
    NS, NROWS, SMALL, BIG, PLAY, DRAW,
    A_DRAW, A_NODRAW, A_FLOCK0, A_NOFLOCK, Game, decode_play,
)


def progress(col):
    """Bandadas que faltan (aprox.) para ganar: el mejor de los dos caminos."""
    dist7 = max(0, 7 - sum(1 for c in col if c > 0))
    need = sorted((3 - c + 1) // 2 if c < 3 else 0 for c in col)
    return min(dist7, need[0] + need[1])


def species_weight(col, s):
    """Cuánto sirve una bandada de s a esta colección (1 = normal)."""
    if col[s] == 0:
        return 1.5
    if col[s] < 3:
        return 1.2
    return 0.6


def hand_value(hand, col):
    v = 0.0
    for s in range(NS):
        k = hand[s]
        if k == 0:
            continue
        if k >= BIG[s]:
            base = 2.2
        elif k >= SMALL[s]:
            base = 1.0 + 0.6 * (k - SMALL[s]) / (BIG[s] - SMALL[s])
        else:
            base = 0.7 * k / SMALL[s]
        v += base * species_weight(col, s)
    return v


def flock_choice(hand, col, legal):
    """Bandada que más acerca a ganar; si no acerca, sólo las grandes."""
    before = progress(col)
    best, best_key = A_NOFLOCK, None
    for a in legal:
        if a == A_NOFLOCK:
            continue
        s = a - A_FLOCK0
        keep = 2 if hand[s] >= BIG[s] else 1
        new = list(col)
        new[s] += keep
        p = progress(new)
        if p >= before and keep == 1:
            continue
        key = (p, -keep, hand[s])
        if best_key is None or key < best_key:
            best, best_key = a, key
    return best


class Bot:
    name = "bot"

    def act(self, g: Game, rng):
        legal = g.legal_actions()
        if g.phase == PLAY:
            return self.play(g, legal, rng)
        if g.phase == DRAW:
            return self.draw(g, legal, rng)
        return self.flock(g, legal, rng)

    def draw(self, g, legal, rng):
        return A_DRAW

    def flock(self, g, legal, rng):
        return flock_choice(g.hands[g.cur], g.cols[g.cur], legal)


class Azar(Bot):
    name = "azar"

    def play(self, g, legal, rng):
        return rng.pick(legal)

    def draw(self, g, legal, rng):
        return rng.pick(legal)

    def flock(self, g, legal, rng):
        return rng.pick(legal)


class Goloso(Bot):
    """Se lleva la mayor cantidad de pájaros; a igualdad, juega menos cartas."""
    name = "goloso"

    def play(self, g, legal, rng):
        h = g.hands[g.cur]
        best, bk = [], None
        for a in legal:
            s, r, side = decode_play(a)
            key = (Game.enclosed(g.rows[r], s, side), -h[s])
            if bk is None or key > bk:
                best, bk = [a], key
            elif key == bk:
                best.append(a)
        return rng.pick(best)

    def flock(self, g, legal, rng):
        h = g.hands[g.cur]
        col = g.cols[g.cur]
        opts = [a for a in legal if a != A_NOFLOCK]
        if not opts:
            return A_NOFLOCK
        return max(opts, key=lambda a: (2 if h[a - A_FLOCK0] >= BIG[a - A_FLOCK0] else 1, col[a - A_FLOCK0] == 0, -a))


DRAW_BONUS = 0.35


class Coleccionista(Bot):
    """Elige la jugada que más mejora el valor de su mano para formar bandadas."""
    name = "coleccionista"

    def play(self, g, legal, rng):
        h = g.hands[g.cur]
        col = g.cols[g.cur]
        base = hand_value(h, col)
        best, bv = [], None
        for a in legal:
            s, r, side = decode_play(a)
            row = g.rows[r]
            k = Game.enclosed(row, s, side)
            nh = list(h)
            nh[s] = 0
            taken = row[:k] if side == 0 else row[len(row) - k:]
            for c in taken:
                nh[c] += 1
            v = hand_value(nh, col) - base + (0.05 * k if k else DRAW_BONUS)
            if bv is None or v > bv + 1e-9:
                best, bv = [a], v
            elif v > bv - 1e-9:
                best.append(a)
        return rng.pick(best)


BOTS = {b.name: b for b in (Azar, Goloso, Coleccionista)}
