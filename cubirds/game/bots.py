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


# Pesos de la evaluación de la mano. AFINADO salió de una búsqueda aleatoria de
# 150 combinaciones contra Coleccionista y Goloso (ver web/engine.js).
DEFAULT = dict(big=2.2, small_base=1.0, slope=0.6, below=0.7, w_new=1.5, w_part=1.2, w_done=0.6, draw=0.35, take=0.05)
AFINADO = dict(big=4.31, small_base=1.37, slope=0.495, below=0.289, w_new=0.946, w_part=3.279, w_done=0.41, draw=0.547, take=0.086)


def species_weight(col, s, P=DEFAULT):
    """Cuánto sirve una bandada de s a esta colección."""
    if col[s] == 0:
        return P["w_new"]
    if col[s] < 3:
        return P["w_part"]
    return P["w_done"]


def hand_value(hand, col, P=DEFAULT):
    v = 0.0
    for s in range(NS):
        k = hand[s]
        if k == 0:
            continue
        if k >= BIG[s]:
            base = P["big"]
        elif k >= SMALL[s]:
            base = P["small_base"] + P["slope"] * (k - SMALL[s]) / (BIG[s] - SMALL[s])
        else:
            base = P["below"] * k / SMALL[s]
        v += base * species_weight(col, s, P)
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


class Coleccionista(Bot):
    """Elige la jugada que más mejora el valor de su mano para formar bandadas."""
    name = "coleccionista"
    P = DEFAULT

    def scores(self, g, legal):
        h = g.hands[g.cur]
        col = g.cols[g.cur]
        P = self.P
        base = hand_value(h, col, P)
        out = []
        for a in legal:
            s, r, side = decode_play(a)
            row = g.rows[r]
            k = Game.enclosed(row, s, side)
            nh = list(h)
            nh[s] = 0
            taken = row[:k] if side == 0 else row[len(row) - k:]
            for c in taken:
                nh[c] += 1
            out.append(hand_value(nh, col, P) - base + (P["take"] * k if k else P["draw"]))
        return out

    def ranked(self, g, legal):
        """Jugadas de mejor a peor (a igual puntaje, en el orden de legal)."""
        sc = self.scores(g, legal)
        return [legal[i] for i in sorted(range(len(legal)), key=lambda i: -sc[i])]

    def play(self, g, legal, rng):
        best, bv = [], None
        for a, v in zip(legal, self.scores(g, legal)):
            if bv is None or v > bv + 1e-9:
                best, bv = [a], v
            elif v > bv - 1e-9:
                best.append(a)
        return rng.pick(best)


class Afinado(Coleccionista):
    """Coleccionista con los pesos que encontró la búsqueda aleatoria."""
    name = "afinado"
    P = AFINADO


# Estratega: lo que dejaron ver los mejores agentes (ver README), con pesos de una
# búsqueda aleatoria de 160 combinaciones contra Afinado y LSTD (ver web/engine.js).
ESTRATEGA = dict(big=3.625, small_base=1.092, slope=0.465, below=0.251, w_new=2.324, w_part=1.977, w_done=0.234, draw=0.419, take=0.077)
RAROS = [0.106, 0.085, 0.085, 0, 0, 0, -0.149, -0.149]  # extra por cada pájaro que se lleva de cada especie
CIERRE = 0.407  # extra por jugar sin llevarse nada y quedar listo para cerrar la ronda, según las manos ajenas
NO_ROBA = 2  # con esta cantidad de cartas o menos no roba: vacía la mano y cierra la ronda


class Estratega(Coleccionista):
    """Sigue los consejos de los mejores agentes: con 2 cartas o menos no roba (cierra la
    ronda y los demás descartan sus manos), forma también bandadas chicas de especies
    nuevas (va a lo ancho, hacia 7 especies) y prefiere llevarse especies raras."""
    name = "estratega"
    P = ESTRATEGA

    def scores(self, g, legal):
        me = g.cur
        h = g.hands[me]
        col = g.cols[me]
        P = self.P
        base = hand_value(h, col, P)
        opp_hand = sum(sum(g.hands[q]) for q in range(g.n) if q != me) / (g.n - 1)
        out = []
        for a in legal:
            s, r, side = decode_play(a)
            row = g.rows[r]
            k = Game.enclosed(row, s, side)
            nh = list(h)
            nh[s] = 0
            taken = row[:k] if side == 0 else row[len(row) - k:]
            for c in taken:
                nh[c] += 1
            v = hand_value(nh, col, P) - base + (P["take"] * k if k else P["draw"])
            for c in taken:
                v += RAROS[c]
            if not k and sum(nh) <= NO_ROBA:
                v += CIERRE * opp_hand / 8
            out.append(v)
        return out

    def draw(self, g, legal, rng):
        return A_NODRAW if sum(g.hands[g.cur]) <= NO_ROBA else A_DRAW

    def flock(self, g, legal, rng):
        h, col = g.hands[g.cur], g.cols[g.cur]
        a = flock_choice(h, col, legal)
        if a != A_NOFLOCK:
            return a
        for a in legal:
            if a != A_NOFLOCK and col[a - A_FLOCK0] == 0:
                return a
        return A_NOFLOCK


def determinize(g, me, rng):
    """Copia de la partida con las cartas ocultas (mazo, descarte y manos ajenas) repartidas al azar."""
    sim = g.clone()
    pool = list(sim.deck) + list(sim.discard)
    for q in range(sim.n):
        if q != me:
            for s in range(NS):
                pool.extend([s] * sim.hands[q][s])
    rng.shuffle(pool)
    pos = 0
    for q in range(sim.n):
        if q != me:
            k = sum(sim.hands[q])
            sim.hands[q] = [0] * NS
            for c in pool[pos:pos + k]:
                sim.hands[q][c] += 1
            pos += k
    nd = len(sim.discard)
    sim.discard = pool[pos:pos + nd]
    sim.deck = pool[pos + nd:]
    return sim


# Pesos de la evaluación del Experto (búsqueda aleatoria, ver web/engine.js).
EXPERTO_W = dict(prog=1.0, hand=0.28, birds=0.51, opp=0.35, win=12.8)


def evaluate(g, p, w=EXPERTO_W):
    """Qué tan bien está p: menos bandadas por formar, mejor mano y más pájaros."""
    if g.done:
        return w["win"] if p in g.winners else -w["win"]
    return -w["prog"] * progress(g.cols[p]) + w["hand"] * hand_value(g.hands[p], g.cols[p], AFINADO) + w["birds"] * sum(g.cols[p])


class Experto(Bot):
    """Monte Carlo: para las mejores jugadas según Afinado, reparte manos posibles
    a los rivales, simula una vuelta de mesa con Afinado y evalúa la posición
    contra el mejor rival."""
    name = "experto"

    def __init__(self, rollouts=8, top=6, horizon=1):
        self.rollouts = rollouts
        self.top = top
        self.horizon = horizon
        self.policy = Afinado()

    def draw(self, g, legal, rng):
        return self.policy.draw(g, legal, rng)

    def play(self, g, legal, rng):
        me = g.cur
        cands = self.policy.ranked(g, legal)[:self.top]
        if len(cands) == 1:
            return cands[0]
        best, bv = cands[0], None
        for a in cands:
            v = 0.0
            for _ in range(self.rollouts):
                sim = determinize(g, me, rng)
                sim.step(a)
                stop = sim.turn + self.horizon * sim.n
                while not sim.done and sim.turn < stop:
                    sim.step(self.policy.act(sim, rng))
                opp = max(evaluate(sim, q) for q in range(sim.n) if q != me)
                v += evaluate(sim, me) - EXPERTO_W["opp"] * opp
            if bv is None or v > bv:
                best, bv = a, v
        return best


BOTS = {b.name: b for b in (Azar, Goloso, Coleccionista, Afinado, Estratega, Experto)}
FAST = ["azar", "goloso", "coleccionista", "afinado", "estratega"]


def make_agent(spec):
    """'goloso', 'experto', 'experto:16:8' (simulaciones por jugada : jugadas candidatas)."""
    kind, _, arg = spec.partition(":")
    if kind == "experto" and arg:
        return Experto(*[int(x) for x in arg.split(":")])
    return BOTS[kind]()
