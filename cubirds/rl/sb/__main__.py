"""
Algoritmos de Sutton & Barto para CuBirds.

    python -m cubirds.rl.sb listar
    python -m cubirds.rl.sb entrenar sarsa q_learning --episodios 20000
    python -m cubirds.rl.sb entrenar --todos --episodios 20000 --exportar

Cada algoritmo juega `episodios` partidas contra una mezcla de bots, aprende y
se evalúa en las mismas mesas que PPO. Con --exportar quedan en la arena
(web/arena/models) como tabla:sb_<algoritmo> o lineal:sb_<algoritmo>.
"""

import argparse
import json
import os
import random
import time
from multiprocessing import Pool

from cubirds.rl.sb import approx, tabular, td_gammon
from cubirds.rl.sb.common import evaluate, key_str, sample_episode, summary

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR = os.path.join(ROOT, "web", "arena", "models")


def registry():
    return {**tabular.ALGORITHMS, **approx.ALGORITHMS, **td_gammon.ALGORITHMS}


def train_one(job):
    name, episodes, seed, out_dir = job
    cls, hp = registry()[name]
    rng = random.Random(seed)
    agent = cls(rng, **hp)
    t0 = time.time()
    curve = []
    make_episode = getattr(agent, "make_episode", sample_episode)
    for e in range(1, episodes + 1):
        agent.play(make_episode(rng))
        if e % max(1, episodes // 10) == 0:
            curve.append(e)
    meta = dict(alg=name, label=cls.label, chapter=cls.chapter, desc=cls.desc, episodes=episodes)
    if hasattr(agent, "table"):
        table = agent.table()
        default = min(table.values()) if table else 0.0  # lo no visto se evita
        values = {key_str(k): round(v, 4) for k, v in table.items()}

        def choose(ep):
            scores = [values.get(key_str(k), default) for k in ep.keys()]
            return scores.index(max(scores))

        model = dict(kind="tabla", default=round(default, 4), values=values)
    else:
        choose = agent.choose
        model = agent.export_data() or dict(kind="solo_python")
    ev = evaluate(choose)
    data = dict(meta, **model, eval=ev, score=round(summary(ev), 3), seconds=round(time.time() - t0, 1))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"sb_{name}.json"), "w") as f:
        json.dump(data, f, ensure_ascii=False)
    return name, data


def export(results):
    os.makedirs(MODELS_DIR, exist_ok=True)
    idx_path = os.path.join(MODELS_DIR, "index.json")
    idx = json.load(open(idx_path)) if os.path.exists(idx_path) else []
    for name, d in results:
        mid = f"sb_{name}"
        if d["kind"] == "solo_python":
            continue
        with open(os.path.join(MODELS_DIR, f"{mid}.json"), "w") as f:
            json.dump({k: v for k, v in d.items() if k not in ("label", "desc", "seconds")}, f,
                      ensure_ascii=False, separators=(",", ":"))
        idx = [m for m in idx if m["id"] != mid]
        idx.append({"id": mid, "file": f"{mid}.json", "kind": d["kind"], "label": f"{d['label']} (S&B {d['chapter']})",
                    "family": "sb", "chapter": d["chapter"], "desc": d["desc"]})
    with open(idx_path, "w") as f:
        json.dump(idx, f, ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser(prog="python -m cubirds.rl.sb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("listar")
    tr = sub.add_parser("entrenar")
    tr.add_argument("algoritmos", nargs="*")
    tr.add_argument("--todos", action="store_true")
    tr.add_argument("--episodios", type=int, default=20000)
    tr.add_argument("--semilla", type=int, default=1)
    tr.add_argument("--procesos", type=int, default=10)
    tr.add_argument("--salida", default=os.path.join(ROOT, "runs", "sb"))
    tr.add_argument("--exportar", action="store_true")
    a = ap.parse_args()
    reg = registry()
    if a.cmd == "listar":
        for name, (cls, hp) in reg.items():
            print(f"{name:22} {cls.chapter:8} {cls.label}  {hp}")
        return
    names = list(reg) if a.todos else a.algoritmos
    jobs = [(n, a.episodios, a.semilla + i, a.salida) for i, n in enumerate(names)]
    with Pool(min(a.procesos, len(jobs))) as pool:
        results = []
        for name, d in pool.imap_unordered(train_one, jobs):
            size = f"{len(d['values'])} claves" if "values" in d else d["kind"]
            print(f"{name:26} índice {d['score']:.3f}  {d['eval']}  ({d['seconds']} s, {size})", flush=True)
            results.append((name, d))
    if a.exportar:
        export(results)
        print(f"Exportados {len(results)} modelos a {MODELS_DIR}")


if __name__ == "__main__":
    main()
