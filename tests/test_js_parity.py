"""El motor JS (web/engine.js) juega exactamente las mismas partidas que el de Python."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from cubirds.game.bots import FAST, make_agent
from cubirds.game.engine import Game

ROOT = Path(__file__).resolve().parent.parent


def py_log(g):
    out = []
    for e in g.log:
        t = e[0]
        if t == "play":
            _, p, s, k, r, side, taken, refill = e
            out.append({"t": t, "p": p, "s": s, "k": k, "r": r, "side": side, "taken": taken, "refill": refill})
        elif t == "draw":
            out.append({"t": t, "p": e[1], "got": e[2]})
        elif t == "flock":
            _, p, s, k, keep = e
            out.append({"t": t, "p": p, "s": s, "k": k, "keep": keep})
        elif t == "round":
            out.append({"t": t, "p": e[1]})
        else:
            out.append({"t": t, "why": e[1], "winners": e[2]})
    return out


def run_parity(jobs):
    res = json.loads(subprocess.run(["node", str(ROOT / "tools/js_games.js"), json.dumps(jobs)],
                                    capture_output=True, text=True, check=True).stdout)
    for (seed, specs), js in zip(jobs, res):
        agents = [make_agent(s) for s in specs]
        g = Game(len(specs), seed)
        while not g.done:
            g.step(agents[g.cur].act(g, g.rng))
        assert py_log(g) == js["log"], f"semilla {seed} {specs}"


@pytest.mark.skipif(not shutil.which("node"), reason="hace falta node")
def test_same_games_in_python_and_js():
    names = FAST
    run_parity([[seed, [names[(seed * 7 + i) % len(names)] for i in range(2 + seed % 4)]] for seed in range(1, 121)])


@pytest.mark.skipif(not shutil.which("node"), reason="hace falta node")
def test_experto_same_in_python_and_js():
    run_parity([[1, ["experto:2:3", "goloso"]], [2, ["coleccionista", "experto:2:4", "goloso"]]])
