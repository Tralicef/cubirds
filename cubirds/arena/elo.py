"""
Elo para partidas de más de dos jugadores.

Cada partida se descompone en enfrentamientos de a pares según el puesto
(0 = ganó, 1 = no ganó): el ganador le gana a cada uno de los demás y los que
no ganaron empatan entre sí (medio punto). La variación de cada jugador es
K / (n - 1) por la suma de (resultado - esperado) contra los demás de la mesa,
así una partida vale lo mismo con 2 o con 5 jugadores. Todas las variaciones
de una partida se calculan con los ratings previos. Dos entradas del mismo
agente en una mesa (eliminatoria) no cuentan como duelo.

La misma fórmula está en web/engine.js (eloUpdate).
"""

INITIAL = 1500.0
K = 32.0


def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update(ratings, seats, ranks, k=K):
    """
    ratings: lista de ratings (se modifica en el lugar)
    seats: índices de agentes en la mesa
    ranks: puesto de cada asiento (menos es mejor)
    Devuelve la variación de cada asiento.
    """
    n = len(seats)
    deltas = [0.0] * n
    for i in range(n):
        for j in range(n):
            if i == j or seats[i] == seats[j]:
                continue
            s = 1.0 if ranks[i] < ranks[j] else 0.5 if ranks[i] == ranks[j] else 0.0
            deltas[i] += s - expected(ratings[seats[i]], ratings[seats[j]])
    deltas = [d * k / (n - 1) for d in deltas]
    for seat, d in zip(seats, deltas):
        ratings[seat] += d
    return deltas
