from cubirds.game.engine import (
    Game, Rng, play_game, play_action, has_won, NS, COUNTS, HAND,
    PLAY, DRAW, FLOCK, A_DRAW, A_NODRAW, A_FLOCK0, A_NOFLOCK,
)
from cubirds.game.bots import BOTS, Azar


def total_cards(g):
    hands = sum(sum(h) for h in g.hands)
    cols = sum(sum(c) for c in g.cols)
    return len(g.deck) + len(g.discard) + sum(len(r) for r in g.rows) + hands + cols


def test_rng_matches_mulberry32():
    # Valores de makeRng(42) en web/engine.js
    r = Rng(42)
    assert [round(r.random(), 10) for _ in range(3)] == [0.6011037519, 0.4482905590, 0.8524657935]


def test_setup():
    g = Game(4, seed=7)
    assert all(len(set(r)) == 3 and len(r) == 3 for r in g.rows)
    assert all(sum(h) == HAND for h in g.hands)
    assert all(sum(c) == 1 for c in g.cols)
    assert total_cards(g) == sum(COUNTS)


def test_enclose_left_and_right():
    g = Game(2, seed=1)
    g.rows[0] = [3, 1, 2, 5]
    g.hands[0] = [0] * NS
    g.hands[0][5] = 2
    g.step(play_action(5, 0, 0))  # dos urracas a la izquierda encierran 3, 1, 2
    # la fila quedó sólo con urracas: se repone del mazo por el mismo lado
    assert g.rows[0][-3:] == [5, 5, 5] and len(set(g.rows[0])) == 2
    assert g.hands[0][3] == g.hands[0][1] == g.hands[0][2] == 1
    assert g.phase == FLOCK  # se llevó pájaros: no puede robar


def test_refill_when_row_has_one_species():
    g = Game(2, seed=3)
    g.rows[1] = [4, 0, 4]
    g.hands[0] = [0] * NS
    g.hands[0][4] = 1
    deck_before = len(g.deck)
    g.step(play_action(4, 1, 1))  # derecha: encierra el 4 de la punta? no, es adyacente
    assert g.phase == DRAW and g.rows[1] == [4, 0, 4, 4]
    g2 = Game(2, seed=3)
    g2.rows[1] = [4, 0, 6]
    g2.hands[0] = [0] * NS
    g2.hands[0][4] = 1
    g2.step(play_action(4, 1, 1))  # encierra 0 y 6, la fila queda sólo con loros
    assert len(set(g2.rows[1])) == 2 and g2.rows[1][:2] == [4, 4]
    assert len(g2.deck) < deck_before


def test_no_enclosure_lets_draw_two():
    g = Game(2, seed=5)
    g.rows[2] = [0, 1, 2]
    g.hands[0] = [0] * NS
    g.hands[0][7] = 2
    g.step(play_action(7, 2, 0))
    assert g.phase == DRAW and g.legal_actions() == [A_NODRAW, A_DRAW]
    g.step(A_DRAW)
    assert sum(g.hands[0]) == 2 and g.phase == FLOCK


def test_flock_small_and_big():
    g = Game(2, seed=9)
    g.phase = FLOCK
    g.hands[0] = [0] * NS
    g.hands[0][0] = 3  # flamencos: chica 2, grande 3
    g.hands[0][1] = 1
    g.cols[0] = [0] * NS
    assert A_FLOCK0 + 0 in g.legal_actions()
    disc = len(g.discard)
    g.step(A_FLOCK0 + 0)
    assert g.cols[0][0] == 2 and len(g.discard) == disc + 1 and g.cur == 1


def test_empty_hand_starts_new_round_same_player():
    g = Game(3, seed=11)
    g.hands[0] = [0] * NS
    g.hands[0][6] = 1
    g.rows[0] = [0, 1, 2]
    g.step(play_action(6, 0, 1))
    g.step(A_NODRAW)
    g.step(A_NOFLOCK)
    assert g.round == 2 and g.cur == 0 and g.phase == PLAY
    assert all(sum(h) == HAND for h in g.hands)


def test_win_conditions():
    assert has_won([1, 1, 1, 1, 1, 1, 1, 0])
    assert has_won([3, 0, 4, 0, 0, 0, 0, 0])
    assert not has_won([3, 2, 1, 1, 1, 1, 0, 0])


def test_full_games_conserve_cards():
    names = list(BOTS)
    for seed in range(200):
        n = 2 + seed % 4
        agents = [BOTS[names[(seed + i) % len(names)]]() for i in range(n)]
        g = Game(n, seed)
        while not g.done:
            assert g.legal_actions()
            g.step(agents[g.cur].act(g, g.rng))
            assert total_cards(g) == sum(COUNTS)
        assert g.winners


def test_deterministic():
    a = play_game([Azar(), Azar(), Azar()], seed=123).log
    b = play_game([Azar(), Azar(), Azar()], seed=123).log
    assert a == b
