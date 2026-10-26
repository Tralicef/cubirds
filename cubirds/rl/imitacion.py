"""
Imitación de un bot (aprendizaje supervisado), para darle a AlphaZero un punto de
partida como el que tuvo AlphaGo con las partidas humanas.

- Datos (tools/imitar_datos.js, en Node y en paralelo): el bot juega mesas de 2 a 5
  y se guarda cada decisión suya con la jugada elegida y quién ganó.
- Red: la misma que PPO y AlphaZero. La política aprende a elegir la jugada del bot
  (entropía cruzada) y el valor a predecir quién ganó (error cuadrático).

Imitando al Estratega la red coincide con él en el 81% de las decisiones y juega
casi igual; AlphaZero a partir de ahí le gana (README, "AlphaZero desde el Estratega").

    python -m cubirds.rl.imitacion --bot estratega --games 4500 --out runs/clon_estratega.json
"""

import argparse
import os
import subprocess

import numpy as np

from cubirds.rl.alphazero import ROOT, load_samples
from cubirds.rl.features import DIM, G_DIM
from cubirds.rl.ppo import HIDDEN, load, save


def gen_data(bot, games, workers, out_dir, seed):
    os.makedirs(out_dir, exist_ok=True)
    procs, files = [], []
    for i in range(workers):
        f = os.path.join(out_dir, f"imit_{seed}_{i}.jsonl")
        files.append(f)
        procs.append(subprocess.Popen(["node", os.path.join(ROOT, "tools", "imitar_datos.js"), bot,
                                       str(games // workers), str(seed * 1000 + i + 1), f], stdout=subprocess.PIPE))
    for p in procs:
        p.communicate()
        if p.returncode:
            raise RuntimeError("fallaron los datos de imitación")
    return files


def fit(files, out, start=None, epochs=12, lr=1e-3, seed=0, meta=None):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    X, PI, GX, Z = load_samples(files)
    N, L = len(X), max(x.shape[0] for x in X)
    Xp = np.zeros((N, L, DIM), np.float32)
    PIp = np.zeros((N, L), np.float32)
    mask = np.zeros((N, L), bool)
    for j, (x, p) in enumerate(zip(X, PI)):
        Xp[j, :len(p)] = x
        PIp[j, :len(p)] = p
        mask[j, :len(p)] = True
    Xp, PIp, mask, G, Zt = map(torch.from_numpy, (Xp, PIp, mask, np.array(GX, np.float32), Z))
    del X, PI, GX

    pi_net = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
    v_net = nn.Sequential(nn.Linear(G_DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
    layers = [(pi_net[0], "W1", "b1"), (pi_net[2], "W2", "b2"), (pi_net[4], "W3", "b3"), (v_net[0], "V1", "c1"), (v_net[2], "V2", "c2")]
    if start:
        P = load(start)
        with torch.no_grad():
            for lyr, w, b in layers:
                lyr.weight.copy_(torch.tensor(P[w].T))
                lyr.bias.copy_(torch.tensor(P[b]))
    perm = torch.randperm(N)
    val, tr = perm[:10000], perm[10000:]
    opt = torch.optim.Adam(list(pi_net.parameters()) + list(v_net.parameters()), lr=lr)
    acc = 0.0
    for ep in range(epochs):
        if ep == 2 * epochs // 3:
            for gr in opt.param_groups:
                gr["lr"] = lr / 3
        order = tr[torch.randperm(len(tr))]
        for b0 in range(0, len(order), 512):
            idx = order[b0:b0 + 512]
            z = pi_net(Xp[idx]).squeeze(-1).masked_fill(~mask[idx], -1e9)
            lp = torch.log_softmax(z, -1).masked_fill(~mask[idx], 0)
            loss = -(PIp[idx] * lp).sum(-1).mean() + ((v_net(G[idx]).squeeze(-1) - Zt[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            z = pi_net(Xp[val]).squeeze(-1).masked_fill(~mask[val], -1e9)
            acc = (z.argmax(-1) == PIp[val].argmax(-1)).float().mean().item()
        print(f"época {ep + 1}: coincide con el bot en {acc:.1%} de las decisiones", flush=True)
    params = {k: v for lyr, w, b in layers for k, v in ((w, lyr.weight.detach().numpy().T.copy()), (b, lyr.bias.detach().numpy().copy()))}
    save(params, out, meta=dict(meta or {}, samples=N, accuracy=round(acc, 4)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="estratega")
    ap.add_argument("--games", type=int, default=4500)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--start", default=os.path.join(ROOT, "web", "arena", "models", "ppo.json"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="runs/clon_estratega.json")
    a = ap.parse_args()
    tmp = os.path.join(os.path.dirname(os.path.abspath(a.out)), "imit_tmp")
    files = gen_data(a.bot, a.games, a.workers, tmp, a.seed)
    fit(files, a.out, start=a.start, epochs=a.epochs, seed=a.seed, meta={"imita": a.bot, "games": a.games})
    for f in files:
        os.remove(f)
