"""Las características de cubirds/rl/features.py y web/engine.js coinciden."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cubirds.game.bots import make_agent
from cubirds.game.engine import Game
from cubirds.rl.features import decision

ROOT = Path(__file__).resolve().parent.parent
JS = """
const E = require('./web/engine.js');
const [seed, n, stops] = JSON.parse(process.argv[1]);
const g = new E.Game(n, seed), bot = E.makeAgent('afinado'), out = [];
let i = 0;
while (!g.done && out.length < stops.length) {
  if (i === stops[out.length]) { const [legal, X] = E.decision(g); out.push([legal, X.map(r => Array.from(r))]); }
  g.step(bot.act(g, g.rng)); i++;
}
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(not shutil.which("node"), reason="hace falta node")
@pytest.mark.parametrize("seed,n", [(3, 2), (8, 3), (21, 4), (40, 5)])
def test_features_match_js(seed, n):
    stops = list(range(0, 120, 3))
    js = json.loads(subprocess.run(["node", "-e", JS, json.dumps([seed, n, stops])], cwd=ROOT,
                                   capture_output=True, text=True, check=True).stdout)
    g, bot, i, k = Game(n, seed), make_agent("afinado"), 0, 0
    while not g.done and k < len(js):
        if i == stops[k]:
            legal, X = decision(g)
            assert legal == js[k][0]
            np.testing.assert_allclose(X, np.array(js[k][1], dtype=np.float32), atol=1e-6)
            k += 1
        g.step(bot.act(g, g.rng))
        i += 1
    assert k > 5


@pytest.mark.skipif(not shutil.which("node"), reason="hace falta node")
def test_net_forward_matches_js(tmp_path):
    from cubirds.rl import ppo
    P = ppo.init_params(3)
    P["W3"] = P["W3"] * 100  # que los logits no sean casi cero
    path = tmp_path / "m.json"
    ppo.save(P, path)
    P = ppo.load(path)
    g, bot = Game(3, 5), make_agent("afinado")
    for _ in range(7):
        g.step(bot.act(g, g.rng))
    legal, X = decision(g)
    py = ppo.logits(P, X)
    js = json.loads(subprocess.run(["node", "-e", """
const E = require('./web/engine.js'); const fs = require('fs');
const m = JSON.parse(fs.readFileSync(process.argv[1]));
const g = new E.Game(3, 5), bot = E.makeAgent('afinado');
for (let i = 0; i < 7; i++) g.step(bot.act(g, g.rng));
const [legal, X] = E.decision(g);
console.log(JSON.stringify(X.map(x => E.netLogit(m.params, x))));
""", str(path)], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    np.testing.assert_allclose(py, js, atol=1e-4)


@pytest.mark.skipif(not shutil.which("node"), reason="hace falta node")
def test_sb_keys_match_js():
    from cubirds.rl.sb.common import key, key_str
    js = json.loads(subprocess.run(["node", "-e", """
const E = require('./web/engine.js');
const out = [];
for (const [seed, n] of [[4, 2], [9, 3], [15, 5]]) {
  const g = new E.Game(n, seed), bot = E.makeAgent('coleccionista');
  for (let i = 0; i < 90 && !g.done; i++) {
    if (g.phase === 'play') out.push(g.legalActions().map(a => E.sbKey(g, a)));
    g.step(bot.act(g, g.rng));
  }
}
console.log(JSON.stringify(out));
"""], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    out = []
    for seed, n in [(4, 2), (9, 3), (15, 5)]:
        g, bot = Game(n, seed), make_agent("coleccionista")
        for _ in range(90):
            if g.done:
                break
            if g.phase == "play":
                out.append([key_str(key(g, a)) for a in g.legal_actions()])
            g.step(bot.act(g, g.rng))
    assert out == js


@pytest.mark.skipif(not shutil.which("node"), reason="hace falta node")
def test_lineal_and_valor_agents_match_js(tmp_path):
    r = np.random.default_rng(1)
    from cubirds.rl.features import DIM
    W1, b1, W2 = r.normal(0, 0.3, (16, DIM)), r.normal(0, 0.1, 16), r.normal(0, 1, 16)
    w = r.normal(0, 1, DIM)
    lin = {"kind": "lineal", "w": np.round(w, 6).tolist(), "b": 0.1}
    val = {"kind": "valor", "W1": np.round(W1.T, 5).tolist(), "b1": np.round(b1, 5).tolist(), "W2": np.round(W2, 5).tolist(), "b2": 0.0}
    (tmp_path / "l.json").write_text(json.dumps(lin))
    (tmp_path / "v.json").write_text(json.dumps(val))
    js = json.loads(subprocess.run(["node", "-e", """
const E = require('./web/engine.js'); const fs = require('fs');
const L = E.makeAgent('lineal:l', { l: JSON.parse(fs.readFileSync(process.argv[1])) });
const V = E.makeAgent('valor:v', { v: JSON.parse(fs.readFileSync(process.argv[2])) });
const out = [], g = new E.Game(4, 12), bot = E.makeAgent('goloso');
while (!g.done && out.length < 30) {
  if (g.phase === 'play') out.push([L.act(g, g.rng), V.act(g, g.rng)]);
  g.step(bot.act(g, g.rng));
}
console.log(JSON.stringify(out));
""", str(tmp_path / "l.json"), str(tmp_path / "v.json")], cwd=ROOT, capture_output=True, text=True, check=True).stdout)
    Wl, bl = np.array(lin["w"]), lin["b"]
    V1, c1, V2 = np.array(val["W1"]).T, np.array(val["b1"]), np.array(val["W2"])
    g, bot, out = Game(4, 12), make_agent("goloso"), []
    while not g.done and len(out) < 30:
        if g.phase == "play":
            legal, X = decision(g)
            X = X.astype(np.float64)
            out.append([legal[int(np.argmax(X @ Wl + bl))], legal[int(np.argmax(np.maximum(X @ V1.T + c1, 0) @ V2))]])
        g.step(bot.act(g, g.rng))
    assert out == js
