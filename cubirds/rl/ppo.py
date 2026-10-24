"""
PPO por jugada.

La política puntúa cada acción legal con una red chica sobre [global, acción]
(cubirds/rl/features.py) y hace softmax entre las legales: la misma red juega,
decide si roba y elige bandadas, en mesas de 2 a 5. El crítico estima la
probabilidad de ganar desde el vector global.

Recompensa: 1 al ganar (repartido si hay empate), 0 si no; sin descuento.
Para que haya señal desde el principio (una red al azar casi nunca gana) se
suma shaping basado en potencial (Ng, Harada y Russell, 1999): cada decisión
recibe Φ(siguiente) − Φ(actual), con Φ = 0,1·pájaros − 0,1·bandadas que faltan
y Φ = 0 al terminar, así la suma de todo el shaping es una constante y la
política óptima no cambia.
Las partidas se juegan en paralelo (multiprocessing) con la red en numpy; la
actualización se hace en torch.

    python -m cubirds.rl.ppo --iters 300 --out runs/ppo
"""

import argparse
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from cubirds.game.bots import make_agent, progress
from cubirds.game.engine import Game, Rng
from cubirds.rl.features import DIM, G_DIM, decision

HIDDEN = 64
OPPONENTS = [("afinado", 0.35), ("coleccionista", 0.2), ("goloso", 0.2), ("experto:4:4", 0.1), ("self", 0.15)]
EVAL_TABLES = [["afinado"], ["afinado", "coleccionista"], ["afinado", "goloso", "coleccionista"], ["afinado", "afinado", "coleccionista", "goloso"]]


def potential(g, p):
    return 0.1 * sum(g.cols[p]) - 0.1 * progress(g.cols[p])


# ---------------- red en numpy (para jugar) ----------------

def init_params(seed=0):
    r = np.random.default_rng(seed)
    def lin(i, o, scale=1.0):
        return (r.normal(0, scale / np.sqrt(i), (i, o)).astype(np.float32), np.zeros(o, np.float32))
    W1, b1 = lin(DIM, HIDDEN)
    W2, b2 = lin(HIDDEN, HIDDEN)
    W3, b3 = lin(HIDDEN, 1, 0.01)
    V1, c1 = lin(G_DIM, HIDDEN)
    V2, c2 = lin(HIDDEN, 1, 0.01)
    return dict(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3, V1=V1, c1=c1, V2=V2, c2=c2)


def logits(P, X):
    h = np.tanh(X @ P["W1"] + P["b1"])
    h = np.tanh(h @ P["W2"] + P["b2"])
    return (h @ P["W3"] + P["b3"])[:, 0]


def value(P, gx):
    h = np.tanh(gx @ P["V1"] + P["c1"])
    return float((h @ P["V2"] + P["c2"])[0])


class NetAgent:
    """Juega con la red: muestrea (entrenando) o elige la mejor (greedy)."""

    def __init__(self, P, greedy=True, record=None):
        self.P, self.greedy, self.record = P, greedy, record

    def act(self, g, rng):
        legal, X = decision(g)
        z = logits(self.P, X)
        if self.greedy:
            return legal[int(np.argmax(z))]
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        u, c, i = rng.random(), 0.0, 0
        for i, pi in enumerate(p):
            c += pi
            if u < c:
                break
        if self.record is not None:
            self.record.append((X, i, float(np.log(p[i] + 1e-12)), value(self.P, X[0, :G_DIM]), potential(g, g.cur)))
        return legal[i]


# ---------------- juego de entrenamiento ----------------

def sample_table(rng):
    names, w = zip(*OPPONENTS)
    k = rng.choice([1, 2, 2, 3, 3, 4])
    return rng.choices(names, w, k=k)


def play_episode(P, past, seed):
    rng = random.Random(seed)
    opp = sample_table(rng)
    n = len(opp) + 1
    me = rng.randrange(n)
    rec = []
    agents, k = [], 0
    for q in range(n):
        if q == me:
            agents.append(NetAgent(P, greedy=False, record=rec))
        else:
            o = opp[k]; k += 1
            agents.append(NetAgent(rng.choice(past), greedy=False) if o == "self" else make_agent(o))
    g = Game(n, seed)
    while not g.done:
        g.step(agents[g.cur].act(g, g.rng))
    reward = (1.0 / len(g.winners)) if me in g.winners else 0.0
    return rec, reward, n


def _worker(args):
    P, past, seeds = args
    return [play_episode(P, past, s) for s in seeds]


def evaluate(P, games=300, seed=777, pool=None):
    """% de victorias en mesas fijas, contra lo justo (1/n)."""
    out = {}
    jobs = [(P, tab, seed + i) for tab in EVAL_TABLES for i in range(games)]
    res = (pool.map(_eval_one, jobs, chunksize=16) if pool else list(map(_eval_one, jobs)))
    for t, tab in enumerate(EVAL_TABLES):
        w = sum(res[t * games:(t + 1) * games]) / games
        out[" + ".join(tab)] = round(w * (len(tab) + 1), 3)  # 1.0 = lo justo
    return out


def _eval_one(args):
    P, tab, seed = args
    n = len(tab) + 1
    me = seed % n
    agents, k = [], 0
    for q in range(n):
        if q == me:
            agents.append(NetAgent(P, greedy=True))
        else:
            agents.append(make_agent(tab[k])); k += 1
    g = Game(n, seed)
    while not g.done:
        g.step(agents[g.cur].act(g, g.rng))
    return (1.0 / len(g.winners)) if me in g.winners else 0.0


# ---------------- PPO en torch ----------------

def train(iters=300, games_per_iter=256, workers=10, out="runs/ppo", lr=3e-4, epochs=4, clip=0.2, lam=0.95,
          ent=0.01, seed=0, eval_every=10):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    os.makedirs(out, exist_ok=True)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.pi = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
            self.v = nn.Sequential(nn.Linear(G_DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
            P = init_params(seed)
            with torch.no_grad():
                for lyr, (w, b) in zip([self.pi[0], self.pi[2], self.pi[4], self.v[0], self.v[2]],
                                       [("W1", "b1"), ("W2", "b2"), ("W3", "b3"), ("V1", "c1"), ("V2", "c2")]):
                    lyr.weight.copy_(torch.tensor(P[w].T)); lyr.bias.copy_(torch.tensor(P[b]))

        def params(self):
            L = [self.pi[0], self.pi[2], self.pi[4], self.v[0], self.v[2]]
            names = [("W1", "b1"), ("W2", "b2"), ("W3", "b3"), ("V1", "c1"), ("V2", "c2")]
            P = {}
            for lyr, (w, b) in zip(L, names):
                P[w] = lyr.weight.detach().numpy().T.copy()
                P[b] = lyr.bias.detach().numpy().copy()
            return P

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    pool = mp.Pool(workers)
    past = [net.params()]
    log = []
    best, best_score = None, -1.0
    rng = random.Random(seed)
    t0 = time.time()
    for it in range(1, iters + 1):
        P = net.params()
        seeds = [rng.randrange(1 << 30) for _ in range(games_per_iter)]
        chunks = [seeds[i::workers] for i in range(workers)]
        eps = [e for part in pool.map(_worker, [(P, past, c) for c in chunks]) for e in part]

        # GAE sin descuento: recompensa sólo al final de la partida
        Xs, acts, oldlp, advs, rets = [], [], [], [], []
        wins = 0.0
        for rec, R, n in eps:
            wins += R * n
            T = len(rec)
            vals = [r[3] for r in rec] + [0.0]
            gae = 0.0
            A = [0.0] * T
            phi = [r[4] for r in rec] + [0.0]
            for t in reversed(range(T)):
                rew = (R if t == T - 1 else 0.0) + phi[t + 1] - phi[t]
                delta = rew + vals[t + 1] - vals[t]
                gae = delta + lam * gae
                A[t] = gae
            for t, (X, i, lp, v, _) in enumerate(rec):
                Xs.append(X); acts.append(i); oldlp.append(lp); advs.append(A[t]); rets.append(A[t] + v)
        N = len(Xs)
        L = max(x.shape[0] for x in Xs)
        Xp = np.zeros((N, L, DIM), np.float32)
        mask = np.zeros((N, L), bool)
        for j, X in enumerate(Xs):
            Xp[j, :X.shape[0]] = X
            mask[j, :X.shape[0]] = True
        Xp, mask = torch.tensor(Xp), torch.tensor(mask)
        acts = torch.tensor(acts)
        oldlp = torch.tensor(oldlp, dtype=torch.float32)
        advs = torch.tensor(advs, dtype=torch.float32)
        rets = torch.tensor(rets, dtype=torch.float32)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        G = Xp[:, 0, :G_DIM]
        for _ in range(epochs):
            perm = torch.randperm(N)
            for b in range(0, N, 1024):
                idx = perm[b:b + 1024]
                z = net.pi(Xp[idx]).squeeze(-1).masked_fill(~mask[idx], -1e9)
                logp_all = torch.log_softmax(z, -1)
                lp = logp_all.gather(1, acts[idx, None]).squeeze(1)
                ratio = torch.exp(lp - oldlp[idx])
                a = advs[idx]
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                probs = logp_all.exp() * mask[idx]
                entropy = -(probs * logp_all.masked_fill(~mask[idx], 0)).sum(-1).mean()
                vl = ((net.v(G[idx]).squeeze(-1) - rets[idx]) ** 2).mean()
                loss = pl + 0.5 * vl - ent * entropy
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        if it % 10 == 0:
            past.append(net.params())
            past = past[-8:]
        row = {"iter": it, "games": it * games_per_iter, "win_index": round(wins / len(eps), 3), "decisions": N,
               "seconds": round(time.time() - t0, 1)}
        if it % eval_every == 0 or it == iters:
            ev = evaluate(net.params(), pool=pool)
            row["eval"] = ev
            score = sum(ev.values()) / len(ev)
            row["eval_mean"] = round(score, 3)
            if score > best_score:
                best_score, best = score, net.params()
                save(best, os.path.join(out, "best.json"), meta={"iter": it, "eval": ev})
        log.append(row)
        print(json.dumps(row), flush=True)
        with open(os.path.join(out, "log.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    save(net.params(), os.path.join(out, "last.json"), meta={"iter": iters})
    pool.close()
    return best


def save(P, path, meta=None):
    """Formato que lee web/engine.js (netAgent): capas como listas [entrada][salida]."""
    with open(path, "w") as f:
        json.dump({"kind": "red", "hidden": HIDDEN, "meta": meta or {},
                   "params": {k: np.round(v, 6).tolist() for k, v in P.items()}}, f)


def load(path):
    with open(path) as f:
        d = json.load(f)
    return {k: np.array(v, dtype=np.float32) for k, v in d["params"].items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--games", type=int, default=256)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ppo")
    a = ap.parse_args()
    train(iters=a.iters, games_per_iter=a.games, workers=a.workers, lr=a.lr, ent=a.ent, seed=a.seed, out=a.out)
