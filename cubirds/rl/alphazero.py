"""
AlphaZero para CuBirds (Silver et al., 2017), adaptado a información imperfecta,
azar y de 2 a 5 jugadores.

- Red: la misma que PPO (cubirds/rl/ppo.py). La política puntúa cada acción legal
  a partir de [estado, acción] y da los priors de la búsqueda; el valor estima la
  probabilidad de ganar desde el estado de un jugador.
- Búsqueda (web/engine.js: azSearch): PUCT sobre un árbol donde juegan todos. En
  cada simulación se reparten manos posibles con las cartas que no ve el que
  piensa (determinización, como ISMCTS), cada nodo guarda el valor de cada
  jugador y el que juega elige según el suyo (max^n). Las hojas se valúan con la
  red, sin simular hasta el final.
- Autojuego (tools/az_selfplay.js, en Node y en paralelo): todos los asientos
  buscan con la misma red, con ruido de Dirichlet en la raíz y temperatura en las
  primeras jugadas.
- Entrenamiento: la política aprende a imitar las visitas de la búsqueda
  (entropía cruzada) y el valor a predecir quién ganó (error cuadrático), con los
  datos de las últimas iteraciones.

A diferencia del AlphaZero original no arranca de cero: parte de los pesos de PPO.

    python -m cubirds.rl.alphazero --iters 8 --games 600 --sims 48 --out runs/az
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np

from cubirds.game.engine import Game
from cubirds.rl.features import DIM, G_DIM, decision
from cubirds.rl.ppo import HIDDEN, load, save

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def selfplay(model_path, games, sims, workers, out_dir, seed, league=None):
    """Lanza `workers` procesos de Node; devuelve la lista de archivos .jsonl."""
    procs, files = [], []
    per = [games // workers + (1 if i < games % workers else 0) for i in range(workers)]
    for i, k in enumerate(per):
        f = os.path.join(out_dir, f"sp_{seed}_{i}.jsonl")
        files.append(f)
        procs.append(subprocess.Popen(["node", os.path.join(ROOT, "tools", "az_selfplay.js"), model_path, str(k),
                                       str(seed * 1000 + i + 1), str(sims), f] + ([league] if league else []),
                                      stdout=subprocess.PIPE))
    for p in procs:
        p.communicate()
        if p.returncode:
            raise RuntimeError("falló el autojuego")
    return files


def load_samples(files):
    X, PI, GX, Z = [], [], [], []
    for f in files:
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                g = Game.from_state(r["s"])
                legal, x = decision(g)
                assert legal == r["legal"], "las acciones legales no coinciden con JS"
                X.append(x)
                PI.append(np.array(r["pi"], np.float32))
                GX.append(x[0, :G_DIM])
                Z.append(r["z"])
    return X, PI, GX, np.array(Z, np.float32)


def evaluate(model_path, specs, games=400):
    out = subprocess.run(["node", os.path.join(ROOT, "tools", "az_eval.js"), str(games), model_path, *specs],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def train(iters=8, games=600, sims=48, workers=6, out="runs/az", start=None, epochs=3, lr=3e-4, window=3,
          value_weight=1.0, seed=1, league=None):
    import torch
    import torch.nn as nn

    os.makedirs(out, exist_ok=True)
    torch.manual_seed(seed)
    start = start or os.path.join(ROOT, "web", "arena", "models", "ppo.json")

    pi_net = nn.Sequential(nn.Linear(DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
    v_net = nn.Sequential(nn.Linear(G_DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
    layers = [(pi_net[0], "W1", "b1"), (pi_net[2], "W2", "b2"), (pi_net[4], "W3", "b3"), (v_net[0], "V1", "c1"), (v_net[2], "V2", "c2")]
    P = load(start)
    with torch.no_grad():
        for lyr, w, b in layers:
            lyr.weight.copy_(torch.tensor(P[w].T))
            lyr.bias.copy_(torch.tensor(P[b]))

    def params():
        return {k: v for lyr, w, b in layers for k, v in ((w, lyr.weight.detach().numpy().T.copy()), (b, lyr.bias.detach().numpy().copy()))}

    opt = torch.optim.Adam(list(pi_net.parameters()) + list(v_net.parameters()), lr=lr)
    cur = os.path.join(out, "it0.json")
    save(params(), cur, meta={"iter": 0, "from": start})
    buffer = []
    log = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time()
    for it in range(1, iters + 1):
        files = selfplay(cur, games, sims, workers, out, seed * 100 + it, league)
        X, PI, GX, Z = load_samples(files)
        for f in files:
            os.remove(f)
        buffer.append((X, PI, GX, Z))
        buffer = buffer[-window:]
        Xs = [x for b in buffer for x in b[0]]
        PIs = [p for b in buffer for p in b[1]]
        G = torch.tensor(np.array([g for b in buffer for g in b[2]]))
        Zt = torch.tensor(np.concatenate([b[3] for b in buffer]))
        N, L = len(Xs), max(x.shape[0] for x in Xs)
        Xp = np.zeros((N, L, DIM), np.float32)
        PIp = np.zeros((N, L), np.float32)
        mask = np.zeros((N, L), bool)
        for j, (x, p) in enumerate(zip(Xs, PIs)):
            Xp[j, :len(p)] = x
            PIp[j, :len(p)] = p
            mask[j, :len(p)] = True
        Xp, PIp, mask = torch.tensor(Xp), torch.tensor(PIp), torch.tensor(mask)
        losses = []
        for _ in range(epochs):
            perm = torch.randperm(N)
            for b0 in range(0, N, 512):
                idx = perm[b0:b0 + 512]
                z = pi_net(Xp[idx]).squeeze(-1).masked_fill(~mask[idx], -1e9)
                lp = torch.log_softmax(z, -1).masked_fill(~mask[idx], 0)
                pl = -(PIp[idx] * lp).sum(-1).mean()
                vl = ((v_net(G[idx]).squeeze(-1) - Zt[idx]) ** 2).mean()
                loss = pl + value_weight * vl
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append((pl.item(), vl.item()))
        cur = os.path.join(out, f"it{it}.json")
        save(params(), cur, meta={"iter": it, "games": it * games, "sims": sims})
        ev = evaluate(cur, ["red:cand"])
        if it % 2 == 0 or it == iters:
            ev.update(evaluate(cur, [f"az:cand:{sims}"], games=200))
        row = {"iter": it, "games": it * games, "samples": N, "policy_loss": round(float(np.mean([l[0] for l in losses[-20:]])), 4),
               "value_loss": round(float(np.mean([l[1] for l in losses[-20:]])), 4), "eval": ev, "seconds": round(time.time() - t0)}
        print(json.dumps(row), flush=True)
        log.write(json.dumps(row) + "\n")
        log.flush()
    return cur


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--sims", type=int, default=48)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/az")
    ap.add_argument("--start", default=None)
    ap.add_argument("--league", default=None, help="agentes de la liga, separados por comas (p. ej. estratega,afinado)")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    train(iters=a.iters, games=a.games, sims=a.sims, workers=a.workers, lr=a.lr, out=a.out, start=a.start,
          league=a.league, window=a.window, seed=a.seed)
