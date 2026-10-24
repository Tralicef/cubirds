"""
Red evolutiva: la misma red por jugada que PPO (cubirds/rl/ppo.py), entrenada
con estrategias evolutivas (Salimans et al., 2017) en vez de gradientes.

En cada generación se prueban 2·pop perturbaciones antitéticas de los pesos
(θ ± σ·ε), cada una juega las mismas partidas (mismas semillas, mesas de 2 a 5
contra bots) y θ se mueve en la dirección de las que ganaron más, usando el
puesto de cada una (fitness shaping por rangos) en vez del puntaje crudo.

    python -m cubirds.rl.neuroevo --gens 150 --out runs/evo
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from cubirds.game.bots import make_agent
from cubirds.game.engine import Game
from cubirds.rl.ppo import NetAgent, evaluate, init_params, save

TABLES = [["afinado"], ["afinado", "coleccionista"], ["afinado", "goloso", "coleccionista"],
          ["afinado", "coleccionista", "goloso", "afinado"], ["coleccionista", "goloso"]]


def flat(P):
    keys = sorted(P)
    return keys, [P[k].shape for k in keys], np.concatenate([P[k].ravel() for k in keys])


def unflat(keys, shapes, v):
    out, i = {}, 0
    for k, sh in zip(keys, shapes):
        n = int(np.prod(sh))
        out[k] = v[i:i + n].reshape(sh).astype(np.float32)
        i += n
    return out


def fitness(args):
    """Índice de victorias (1 = lo justo) en las partidas de esta generación."""
    P, seeds = args
    agent = NetAgent(P, greedy=True)
    total = 0.0
    for s in seeds:
        tab = TABLES[s % len(TABLES)]
        n = len(tab) + 1
        me = (s // len(TABLES)) % n
        it = iter(tab)
        agents = [agent if q == me else make_agent(next(it)) for q in range(n)]
        g = Game(n, s)
        while not g.done:
            g.step(agents[g.cur].act(g, g.rng))
        total += (n / len(g.winners)) if me in g.winners else 0.0
    return total / len(seeds)


def train(gens=150, pop=24, games=20, sigma=0.05, lr=0.03, workers=10, out="runs/evo", seed=0, eval_every=10):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    keys, shapes, theta = flat(init_params(seed))
    # que la salida no arranque casi en cero: si no, todas las perturbaciones juegan igual
    P0 = unflat(keys, shapes, theta)
    P0["W3"] *= 30
    keys, shapes, theta = flat(P0)
    pool = mp.Pool(workers)
    best, best_score = None, -1.0
    t0 = time.time()
    rs = random.Random(seed)
    for gen in range(1, gens + 1):
        eps = rng.normal(0, 1, (pop, theta.size)).astype(np.float32)
        seeds = [rs.randrange(1 << 30) for _ in range(games)]
        cands = [theta + sigma * e for e in eps] + [theta - sigma * e for e in eps]
        fit = np.array(pool.map(fitness, [(unflat(keys, shapes, c), seeds) for c in cands]))
        ranks = np.empty(len(fit))
        ranks[np.argsort(fit)] = np.arange(len(fit))
        u = ranks / (len(fit) - 1) - 0.5  # entre -0.5 y 0.5
        grad = ((u[:pop] - u[pop:])[:, None] * eps).sum(0) / (2 * pop * sigma)
        theta = theta + lr * grad
        row = {"gen": gen, "games": gen * 2 * pop * games, "fit_mean": round(float(fit.mean()), 3),
               "fit_max": round(float(fit.max()), 3), "seconds": round(time.time() - t0, 1)}
        if gen % eval_every == 0 or gen == gens:
            P = unflat(keys, shapes, theta)
            ev = evaluate(P, pool=pool)
            row["eval"] = ev
            score = sum(ev.values()) / len(ev)
            row["eval_mean"] = round(score, 3)
            if score > best_score:
                best_score, best = score, P
                save(best, os.path.join(out, "best.json"), meta={"gen": gen, "eval": ev})
        print(json.dumps(row), flush=True)
        with open(os.path.join(out, "log.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    pool.close()
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="runs/evo")
    a = ap.parse_args()
    train(gens=a.gens, pop=a.pop, games=a.games, sigma=a.sigma, lr=a.lr, workers=a.workers, out=a.out)
