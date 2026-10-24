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
