"""
Reglas de CuBirds.

La misma lógica está en web/engine.js, con el mismo generador de números
aleatorios (mulberry32): con la misma semilla y los mismos agentes, una
partida en Python y otra en el navegador son idénticas carta por carta.

El turno se parte en hasta tres decisiones, todas en un único espacio de
acciones (útil para los agentes de RL):

  0..63   JUGAR: especie * 8 + fila * 2 + lado (0 = izquierda, 1 = derecha)
  64, 65  ROBAR: 64 = no robar, 65 = robar 2 (sólo si no encerró pájaros)
  66..73  BANDADA con la especie (acción - 66); 74 = no formar bandada
"""

SPECIES = ["Flamenco", "Búho", "Tucán", "Pato", "Loro", "Urraca", "Carricero", "Petirrojo"]
COUNTS = [7, 10, 10, 13, 13, 17, 20, 20]
SMALL = [2, 3, 3, 4, 4, 5, 6, 6]
BIG = [3, 4, 4, 6, 6, 7, 9, 9]
NS = 8
NROWS = 4
HAND = 8

PLAY, DRAW, FLOCK = "play", "draw", "flock"
A_NODRAW, A_DRAW = 64, 65
A_FLOCK0, A_NOFLOCK = 66, 74
NACTIONS = 75

MAX_TURNS = 1000  # tope de seguridad: si se alcanza, gana el que más pájaros tenga


def play_action(s, r, side):
    return s * 8 + r * 2 + side


def decode_play(a):
    return a // 8, (a % 8) // 2, a % 2


def _imul(a, b):
    return (a * b) & 0xFFFFFFFF


class Rng:
    """mulberry32, igual que makeRng en web/engine.js."""

    def __init__(self, seed):
        self.a = (seed & 0xFFFFFFFF) or 1

    def random(self):
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        a = self.a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    def int(self, n):
        return int(self.random() * n)

    def pick(self, seq):
        return seq[self.int(len(seq))]

    def shuffle(self, arr):
        for i in range(len(arr) - 1, 0, -1):
            j = self.int(i + 1)
            arr[i], arr[j] = arr[j], arr[i]
        return arr


def has_won(col):
    """7 especies distintas, o 2 especies con al menos 3 pájaros cada una."""
    return sum(1 for c in col if c > 0) >= 7 or sum(1 for c in col if c >= 3) >= 2


class Game:
    def __init__(self, nplayers, seed=0):
        assert 2 <= nplayers <= 5
        self.n = nplayers
        self.rng = Rng(seed)
        deck = [s for s in range(NS) for _ in range(COUNTS[s])]
        self.deck = self.rng.shuffle(deck)
        self.discard = []
        self.rows = []
        for _ in range(NROWS):
            row = []
            while len(row) < 3:
                c = self.deck.pop()
                if c in row:
                    self.discard.append(c)
                else:
                    row.append(c)
            self.rows.append(row)
        self.hands = [[0] * NS for _ in range(nplayers)]
        for p in range(nplayers):
            for _ in range(HAND):
                self.hands[p][self.deck.pop()] += 1
        self.cols = [[0] * NS for _ in range(nplayers)]
        for p in range(nplayers):
            self.cols[p][self.deck.pop()] += 1
        self.cur = 0
        self.phase = PLAY
        self.turn = 0
        self.round = 1
        self.done = False
        self.winners = []
        self.log = []  # eventos de la partida, para la interfaz y los tests

    @classmethod
    def from_state(cls, d):
        """Estado compacto (como lo escribe tools/az_selfplay.js) -> partida con lo necesario para
        las acciones legales y las características; el mazo sólo conserva su tamaño."""
        g = cls.__new__(cls)
        g.n = d["n"]
        g.rng = Rng(1)
        g.deck = [0] * d["deck"]
        g.discard = []
        g.rows = [list(r) for r in d["rows"]]
        g.hands = [list(h) for h in d["hands"]]
        g.cols = [list(c) for c in d["cols"]]
        g.cur = d["cur"]
        g.phase = d["phase"]
        g.turn = 0
        g.round = 1
        g.done = False
        g.winners = []
        g.log = []
        return g

    def clone(self):
        g = Game.__new__(Game)
        g.__dict__.update(self.__dict__)
        g.deck = list(self.deck)
        g.discard = list(self.discard)
        g.rows = [list(r) for r in self.rows]
        g.hands = [list(h) for h in self.hands]
        g.cols = [list(c) for c in self.cols]
        g.winners = list(self.winners)
        g.log = []
        return g

    # ---------- mazo ----------
    def _draw(self):
        if not self.deck:
            if not self.discard:
                return None
            self.deck = self.rng.shuffle(self.discard)
            self.discard = []
        return self.deck.pop()

    # ---------- acciones legales ----------
    def legal_actions(self):
        if self.done:
            return []
        h = self.hands[self.cur]
        if self.phase == PLAY:
            return [play_action(s, r, side) for s in range(NS) if h[s] for r in range(NROWS) for side in (0, 1)]
        if self.phase == DRAW:
            return [A_NODRAW, A_DRAW]
        return [A_FLOCK0 + s for s in range(NS) if h[s] >= SMALL[s]] + [A_NOFLOCK]

    @staticmethod
    def enclosed(row, s, side):
        """Cuántas cartas encierra jugar la especie s en ese lado de la fila."""
        idx = range(len(row)) if side == 0 else range(len(row) - 1, -1, -1)
        for k, i in enumerate(idx):
            if row[i] == s:
                return k
        return 0

    # ---------- avanzar ----------
    def step(self, a):
        p = self.cur
        h = self.hands[p]
        if self.phase == PLAY:
            s, r, side = decode_play(a)
            k = h[s]
            assert k > 0, "especie que no está en la mano"
            h[s] = 0
            row = self.rows[r]
            taken = []
            for i in range(len(row)):
                j = i if side == 0 else len(row) - 1 - i
                if row[j] == s:
                    taken = row[:i] if side == 0 else row[len(row) - i:]
                    break
            if side == 0:
                row = [s] * k + row[len(taken):]
            else:
                row = row[:len(row) - len(taken)] + [s] * k
            for c in taken:
                h[c] += 1
            refill = []
            if taken and len(set(row)) == 1:
                while len(set(row)) == 1:
                    c = self._draw()
                    if c is None:
                        break
                    refill.append(c)
                    row = row + [c] if side == 1 else [c] + row
            self.rows[r] = row
            self.log.append(("play", p, s, k, r, side, list(taken), refill))
            self.phase = FLOCK if taken else DRAW
        elif self.phase == DRAW:
            got = []
            if a == A_DRAW:
                for _ in range(2):
                    c = self._draw()
                    if c is not None:
                        h[c] += 1
                        got.append(c)
            self.log.append(("draw", p, got))
            self.phase = FLOCK
        else:
            if a != A_NOFLOCK:
                s = a - A_FLOCK0
                k = h[s]
                assert k >= SMALL[s], "no alcanza para una bandada"
                keep = 2 if k >= BIG[s] else 1
                h[s] = 0
                self.cols[p][s] += keep
                self.discard.extend([s] * (k - keep))
                self.log.append(("flock", p, s, k, keep))
            self._end_turn()

    def _end_turn(self):
        p = self.cur
        self.turn += 1
        if has_won(self.cols[p]):
            self._finish([p], "win")
            return
        if sum(self.hands[p]) == 0:
            # Fin de la ronda: los demás descartan y el que vació reparte y sigue jugando.
            for q in range(self.n):
                for s in range(NS):
                    self.discard.extend([s] * self.hands[q][s])
                    self.hands[q][s] = 0
            if len(self.deck) < HAND * self.n:
                self.deck = self.rng.shuffle(self.discard) + self.deck
                self.discard = []
            if len(self.deck) < HAND * self.n:
                self._finish_by_count("deck")
                return
            for q in [(p + i) % self.n for i in range(self.n)]:
                for _ in range(HAND):
                    self.hands[q][self.deck.pop()] += 1
            self.round += 1
            self.log.append(("round", p))
        else:
            self.cur = (p + 1) % self.n
        if self.turn >= MAX_TURNS:
            self._finish_by_count("cap")
            return
        self.phase = PLAY

    def _finish_by_count(self, why):
        tot = [sum(c) for c in self.cols]
        m = max(tot)
        self._finish([q for q in range(self.n) if tot[q] == m], why)

    def _finish(self, winners, why):
        self.done = True
        self.winners = winners
        self.end_reason = why
        self.log.append(("end", why, list(winners)))

    # ---------- resultado ----------
    def ranks(self):
        """Puesto de cada jugador (0 = ganó); los que no ganaron empatan entre sí."""
        return [0 if q in self.winners else 1 for q in range(self.n)]


def play_game(agents, seed=0, game=None):
    """Juega una partida completa; agents[i].act(game, rng) -> acción."""
    g = game or Game(len(agents), seed)
    while not g.done:
        g.step(agents[g.cur].act(g, g.rng))
    return g
