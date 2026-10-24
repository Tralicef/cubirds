/*
 * Laboratorio: entrena algoritmos de RL en vivo en un Web Worker y le manda a
 * la página lo necesario para mostrar cómo aprenden.
 *
 * Usa la misma idea que cubirds/rl/sb: valores de afterstate (Sutton & Barto
 * 6.8). Cada jugada posible se resume en una clave (cuántos pájaros se lleva,
 * qué bandada podría formar después, cuánto le falta para ganar, ...) o en las
 * 65 características por jugada, y se aprende cuánto vale. Las claves y
 * características vienen de engine.js, así que coinciden con las de Python.
 * El agente decide la jugada; robar y formar bandadas lo hace como Afinado.
 *
 * Recompensa: 1 al ganar la partida y, con shaping, entre una jugada y la
 * siguiente el cambio de Φ = 0,1·pájaros − 0,1·bandadas que faltan. Los métodos
 * tabulares que miran hasta el final aprenden sin shaping (la probabilidad de
 * ganar): con shaping, el Φ del estado previo, que la clave no ve, ensucia el
 * valor de cada clave (ver cubirds/rl/sb/common.py).
 */
importScripts('engine.js');
const E = self.CuBirdsEngine;

// ---------------- partida desde un asiento (como cubirds/rl/sb/common.py: Episode) ----------------
const TRAIN_OPP = [['afinado', 0.35], ['coleccionista', 0.25], ['goloso', 0.25], ['azar', 0.15]];
const TABLE_OPP = [1, 2, 2, 3, 3, 4];
const EVAL_TABLES = [['afinado', 'coleccionista'], ['afinado', 'goloso', 'coleccionista']];
const BOTS = {};
const botOf = s => BOTS[s] || (BOTS[s] = E.makeAgent(s));
const sum = a => a.reduce((x, y) => x + y, 0);
const potential = (g, p) => 0.1 * sum(g.cols[p]) - 0.1 * E.progress(g.cols[p]);

class Episode {
  constructor(opps, rng, shaping = true) {
    this.rng = rng;
    this.shaping = shaping;
    this.n = opps.length + 1;
    this.me = rng.int(this.n);
    this.g = new E.Game(this.n, rng.int(2 ** 30) + 1);
    let k = 0;
    this.agents = Array.from({ length: this.n }, (_, q) => q === this.me ? null : (typeof opps[k] === 'string' ? botOf(opps[k++]) : opps[k++]));
    this.advance();
    this.phi = potential(this.g, this.me);
  }
  advance() {
    const g = this.g;
    while (!g.done && !(g.cur === this.me && g.phase === 'play')) g.step((g.cur === this.me ? botOf('afinado') : this.agents[g.cur]).act(g, g.rng));
  }
  get done() { return this.g.done; }
  legal() { return this.g.legalActions(); }
  keys() { return this.legal().map(a => E.sbKey(this.g, a)); }
  feats() { return E.decision(this.g)[1].map(x => Array.from(x)); }
  step(i) {
    const g = this.g;
    g.step(g.legalActions()[i]);
    this.advance();
    if (g.done) return (g.winners.includes(this.me) ? 1 / g.winners.length : 0) - (this.shaping ? this.phi : 0);
    if (!this.shaping) return 0;
    const phi = potential(g, this.me), r = phi - this.phi;
    this.phi = phi;
    return r;
  }
  // índice de victorias: 1 = lo justo (ganar 1 de cada n)
  result() { return this.g.winners.includes(this.me) ? this.n / this.g.winners.length : 0; }
}
function trainEpisode(rng, shaping = true) {
  const names = TRAIN_OPP.map(x => x[0]), w = TRAIN_OPP.map(x => x[1]);
  const pick = () => { let u = rng() * sum(w); for (let i = 0; i < w.length; i++) { u -= w[i]; if (u <= 0) return names[i]; } return names[0]; };
  const k = TABLE_OPP[rng.int(TABLE_OPP.length)];
  return new Episode(Array.from({ length: k }, pick), rng, shaping);
}
// Evaluación greedy en mesas y partidas fijas (siempre las mismas semillas).
function evaluate(chooser, games = 120) {
  let total = 0;
  for (const tab of EVAL_TABLES) for (let k = 0; k < games; k++) {
    const ep = new Episode(tab, E.makeRng(77000 + k));
    while (!ep.done) ep.step(chooser(ep));
    total += ep.result();
  }
  return total / (EVAL_TABLES.length * games);
}
const sample = (p, rng) => { let r = rng(), j = 0; for (; j < p.length - 1; j++) { r -= p[j]; if (r <= 0) break; } return j; };
const softmax = h => { const m = Math.max(...h), e = h.map(v => Math.exp(v - m)), s = sum(e); return e.map(v => v / s); };
const argmax = v => { let b = 0; v.forEach((x, i) => { if (x > v[b]) b = i; }); return b; };

// ---------------- algoritmos tabulares ----------------
const AT_END = ['mc', 'mc_off', 'sarsa_n', 'sarsa_lambda', 'tree_backup', 'q_lambda'];
function Tabular(hp, rng, kind) {
  const Q = new Map(), N = new Map(), Q2 = new Map(), C = new Map();
  const q1 = k => Q.has(k) ? Q.get(k) : (kind === 'optimista' ? hp.init : 0);
  const q = kind === 'double_q' ? k => (q1(k) + (Q2.get(k) || 0)) / 2 : q1;
  const greedy = ks => { let b = 0, bv = -Infinity; ks.forEach((k, i) => { const v = q(k); if (v > bv) { bv = v; b = i; } }); return b; };
  const epsProbs = ks => { const n = ks.length, p = new Array(n).fill(hp.epsilon / n); p[greedy(ks)] += 1 - hp.epsilon; return p; };
  let t = 0;
  const visit = k => N.set(k, (N.get(k) || 0) + 1);
  const setQ = (k, v) => Q.set(k, v);
  const maxQ = ks => Math.max(...ks.map(q));

  function explore(ks) {
    if (kind === 'ucb') {
      for (let i = 0; i < ks.length; i++) if (!N.get(ks[i])) return { j: i, why: 'situación nunca probada' };
      const lt = Math.log(t + 1);
      let b = 0, bv = -Infinity;
      ks.forEach((k, i) => { const v = q(k) + hp.c * Math.sqrt(lt / N.get(k)); if (v > bv) { bv = v; b = i; } });
      return { j: b, why: 'UCB: valor + bono de exploración' };
    }
    if (rng() < hp.epsilon) return { j: rng.int(ks.length), why: `explora (ε = ${hp.epsilon})` };
    return { j: greedy(ks), why: 'la de mayor valor estimado' };
  }

  // Devuelve la descripción de la actualización para el modo paso a paso.
  function update(step, next) {
    const { ks, j, r } = step, k = ks[j], old = q(k);
    let target, how;
    if (kind === 'bandido' || kind === 'ucb') {
      visit(k); setQ(k, old + (r - old) / N.get(k));
      return { k, old, target: r, newV: q(k), how: 'promedio de lo que ganó en esta jugada' };
    }
    if (kind === 'optimista') {
      visit(k); setQ(k, old + hp.alpha * (r - old));
      return { k, old, target: r, newV: q(k), how: `paso fijo α; arrancó en ${hp.init}, como si cada jugada fuera buenísima` };
    }
    if (AT_END.includes(kind)) return null; // al final de la partida
    if (kind === 'double_q') {
      // dos tablas: una elige la mejor jugada siguiente y la otra la evalúa
      const swap = rng() < 0.5, A = swap ? Q2 : Q, B = swap ? Q : Q2, qa = x => A.get(x) || 0, qb = x => B.get(x) || 0;
      if (next) { let b = 0; next.ks.forEach((x, i) => { if (qa(x) > qa(next.ks[b])) b = i; }); target = r + qb(next.ks[b]); }
      else target = r;
      A.set(k, qa(k) + hp.alpha * (target - qa(k))); visit(k);
      return { k, old, target, newV: q(k), how: `actualiza Q${swap ? 2 : 1}: elige con Q${swap ? 2 : 1} y evalúa con Q${swap ? 1 : 2}` };
    }
    if (!next) { target = r; how = 'última jugada: sólo la recompensa'; }
    else if (kind === 'sarsa') { target = r + q(next.ks[next.j]); how = 'recompensa + V(jugada que hace después)'; }
    else if (kind === 'q_learning' || kind === 'dyna_q' || kind === 'prioritized') { target = r + maxQ(next.ks); how = 'recompensa + V(mejor jugada después)'; }
    else if (kind === 'expected_sarsa') { const p = epsProbs(next.ks); target = r + next.ks.reduce((a, x, i) => a + p[i] * q(x), 0); how = 'recompensa + V esperado después (ε-greedy)'; }
    visit(k);
    setQ(k, old + hp.alpha * (target - old));
    if (kind === 'dyna_q') plan(k, r, next ? next.ks : null);
    if (kind === 'prioritized') sweep(k, r, next ? next.ks : null);
    return { k, old, target, newV: q(k), how };
  }

  const model = new Map(), visited = [];
  function remember(k, r, ks1) {
    if (!model.has(k)) { model.set(k, []); visited.push(k); }
    const b = model.get(k); if (b.length < 30) b.push([r, ks1]); else b[rng.int(30)] = [r, ks1];
  }
  function plan(k, r, ks1) {
    remember(k, r, ks1);
    for (let i = 0; i < hp.planning; i++) {
      const kk = visited[rng.int(visited.length)], [c, nx] = model.get(kk)[rng.int(model.get(kk).length)];
      const tg = nx ? c + maxQ(nx) : c;
      setQ(kk, q(kk) + hp.alpha * (tg - q(kk)));
    }
  }

  // Barrido priorizado (S&B 8.4): repasa primero las situaciones cuyo valor más cambiaría,
  // y cuando una cambia, anota a las que llevan a ella (sus predecesoras).
  const pred = new Map(), PQ = new Map();
  const expTarget = kk => { const b = model.get(kk), m = Math.min(b.length, 8); let s = 0; for (let i = 0; i < m; i++) { const [c, nx] = b[i]; s += nx ? c + maxQ(nx) : c; } return s / m; };
  const push = kk => { // cola acotada para que no se vuelva lenta
    if (PQ.size >= 200 && !PQ.has(kk)) return;
    const P = Math.abs(expTarget(kk) - q(kk)); if (P > hp.theta) PQ.set(kk, Math.max(P, PQ.get(kk) || 0));
  };
  function sweep(k, r, ks1) {
    remember(k, r, ks1);
    if (ks1) for (const x of ks1) { if (!pred.has(x)) pred.set(x, { s: new Set(), a: [] }); const p = pred.get(x); if (!p.s.has(k)) { p.s.add(k); p.a.push(k); } }
    push(k);
    for (let i = 0; i < hp.planning && PQ.size; i++) {
      let top = null, tp = -1;
      for (const [x, p] of PQ) if (p > tp) { tp = p; top = x; }
      PQ.delete(top);
      setQ(top, q(top) + hp.alpha * (expTarget(top) - q(top)));
      const ps = pred.get(top);
      if (ps) for (let m = 0; m < Math.min(3, ps.a.length); m++) push(ps.a[rng.int(ps.a.length)]);
    }
  }

  function endEpisode(traj) {
    const T = traj.length;
    if (kind === 'mc_off') {
      // off-policy con muestreo de importancia ponderado: aprende la política greedy jugando ε-greedy
      let G = 0, W = 1;
      for (let i = T - 1; i >= 0; i--) {
        const s = traj[i], k = s.ks[s.j];
        G += s.r; visit(k);
        C.set(k, (C.get(k) || 0) + W); setQ(k, q(k) + W / C.get(k) * (G - q(k)));
        if (s.j !== s.g) break; // la greedy no habría jugado esto: lo anterior no sirve
        W /= 1 - hp.epsilon + hp.epsilon / s.ks.length;
      }
    } else if (kind === 'tree_backup') {
      // n pasos de árbol: sigue lo jugado mientras coincida con la greedy; si no, corta con Q
      const tb = (i, d) => {
        const c = traj[i].r; if (i + 1 >= T) return c;
        const nx = traj[i + 1], g = nx.g;
        return d >= hp.n - 1 || nx.j !== g ? c + q(nx.ks[g]) : c + tb(i + 1, d + 1);
      };
      for (let i = 0; i < T; i++) { const k = traj[i].ks[traj[i].j], G = tb(i, 0); visit(k); setQ(k, q(k) + hp.alpha * (G - q(k))); }
    } else if (kind === 'q_lambda') {
      // Watkins: trazas como SARSA(λ) con objetivo de Q-learning; se borran al explorar
      const z = new Map();
      traj.forEach((s, i) => {
        const k = s.ks[s.j], nx = traj[i + 1];
        const delta = s.r + (nx ? maxQ(nx.ks) : 0) - q(k);
        z.set(k, 1); visit(k);
        for (const [x, zx] of z) setQ(x, q(x) + hp.alpha * delta * zx);
        if (nx && nx.j === nx.g) for (const [x, zx] of z) z.set(x, zx * hp.lambda); else z.clear();
      });
    } else if (kind === 'mc') {
      let g = 0;
      for (let i = T - 1; i >= 0; i--) { g += traj[i].r; const k = traj[i].ks[traj[i].j]; visit(k); setQ(k, q(k) + (g - q(k)) / N.get(k)); }
    } else if (kind === 'sarsa_n') {
      for (let i = 0; i < T; i++) {
        let g = 0; for (let m = i; m < Math.min(i + hp.n, T); m++) g += traj[m].r;
        if (i + hp.n < T) g += q(traj[i + hp.n].ks[traj[i + hp.n].j]);
        const k = traj[i].ks[traj[i].j]; visit(k); setQ(k, q(k) + hp.alpha * (g - q(k)));
      }
    } else if (kind === 'sarsa_lambda') {
      const z = new Map();
      traj.forEach((s, i) => {
        const k = s.ks[s.j], nx = i + 1 < T ? q(traj[i + 1].ks[traj[i + 1].j]) : 0;
        const delta = s.r + nx - q(k);
        z.set(k, 1); visit(k);
        for (const [x, zx] of z) { setQ(x, q(x) + hp.alpha * delta * zx); z.set(x, zx * hp.lambda); }
      });
    }
  }

  return {
    type: 'tabla', shaping: ['bandido', 'ucb', 'optimista'].includes(kind),
    act(ep) { const ks = ep.keys(); t++; return { ks, g: greedy(ks), ...explore(ks) }; },
    greedyIndex(ep) { return greedy(ep.keys()); },
    update, endEpisode, q, N: k => N.get(k) || 0,
    view() { return heatView(N, q); },
  };
}

// Mapa de calor: pájaros que se lleva (filas) × mejor bandada que podría formar después (columnas),
// promedio ponderado por visitas.
function heatView(N, val, unit) {
  const cells = [0, 1, 2, 3, 4].map(() => [0, 1, 2].map(() => ({ s: 0, n: 0 })));
  for (const [k, n] of N) {
    if (!n) continue;
    const [taken, ready] = k.split(',').map(Number);
    cells[taken][ready].s += val(k) * n; cells[taken][ready].n += n;
  }
  return { kind: 'heat', unit, cells: cells.map(row => row.map(c => c.n ? [c.s / c.n, c.n] : null)), states: N.size };
}

// ---------------- más tabulares ----------------
// Bandido por gradiente: preferencias H por situación, softmax y línea de base.
function GradBandit(hp, rng) {
  const H = new Map(), N = new Map();
  let base = 0, steps = 0;
  const h = k => H.get(k) || 0;
  const probs = ks => softmax(ks.map(h));
  return {
    type: 'tabla',
    act(ep) { const ks = ep.keys(), p = probs(ks), j = sample(p, rng); return { ks, p, j, why: `sorteo con probabilidad ${(p[j] * 100).toFixed(0)}%` }; },
    greedyIndex(ep) { return argmax(ep.keys().map(h)); },
    update(step) {
      const { ks, j, r } = step, p = probs(ks), k = ks[j], old = h(k);
      steps++; const prev = base, delta = r - base; base += (r - base) / steps;
      ks.forEach((x, i) => H.set(x, h(x) + hp.alpha * delta * ((i === j ? 1 : 0) - p[i])));
      N.set(k, (N.get(k) || 0) + 1);
      return { k, text: `preferencia ${old.toFixed(2)} → ${h(k).toFixed(2)}   (ganó ${r.toFixed(2)}; el promedio venía en ${prev.toFixed(2)}: le fue ${delta >= 0 ? 'mejor' : 'peor'} que lo habitual, así que ${delta >= 0 ? 'sube' : 'baja'} esta situación y ${delta >= 0 ? 'bajan' : 'suben'} las otras)` };
    },
    endEpisode() {},
    view() { return heatView(N, h, 'preferencia'); },
  };
}

// ---------------- métodos lineales (65 características por jugada) ----------------
const D = E.DIM, GD = E.G_DIM;
const dotv = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
function Linear(hp, rng, kind) {
  let w = new Array(D).fill(0), b = 0;
  const v = x => dotv(x, w) + b;
  const A = [], bv = new Array(D + 1).fill(0); // LSTD
  for (let i = 0; i <= D; i++) { A.push(new Array(D + 1).fill(0)); A[i][i] = 1; }
  let episodes = 0;
  const sgd = (x, tgt, a = hp.alpha) => { const d = tgt - v(x); for (let i = 0; i < D; i++) w[i] += a * d * x[i]; b += a * d; return d; };
  function solve() { // eliminación gaussiana sobre A·θ = bv
    const n = D + 1, M = A.map((r, i) => r.concat(bv[i]));
    for (let c = 0; c < n; c++) {
      let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]]; if (Math.abs(M[c][c]) < 1e-9) continue;
      for (let r = 0; r < n; r++) if (r !== c) { const f = M[r][c] / M[c][c]; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
    }
    const th = M.map((r, i) => Math.abs(r[i]) < 1e-9 ? 0 : r[n] / r[i]);
    w = th.slice(0, D); b = th[D];
  }
  return {
    type: 'pesos',
    act(ep) {
      const X = ep.feats(), ks = ep.keys(), means = X.map(v);
      if (rng() < hp.epsilon) return { X, ks, means, j: rng.int(X.length), why: `explora (ε = ${hp.epsilon})` };
      return { X, ks, means, j: argmax(means), why: 'la de mayor valor estimado' };
    },
    greedyIndex(ep) { return argmax(ep.feats().map(v)); },
    update(step, next) {
      const x = step.X[step.j], old = v(x);
      if (kind === 'bandido_lineal') { sgd(x, step.r); return { k: step.ks[step.j], old, target: step.r, newV: v(x), how: 'regresión de lo que ganó en esta jugada' }; }
      if (kind === 'sarsa_lin' || kind === 'q_lin') {
        const tgt = !next ? step.r : kind === 'sarsa_lin' ? step.r + v(next.X[next.j]) : step.r + Math.max(...next.X.map(v));
        sgd(x, tgt);
        return { k: step.ks[step.j], old, target: tgt, newV: v(x), how: kind === 'sarsa_lin' ? 'recompensa + v̂(jugada que hace después)' : 'recompensa + máx v̂(turno siguiente)' };
      }
      return null; // los demás aprenden al terminar la partida
    },
    endEpisode(traj) {
      const T = traj.length;
      if (kind === 'mc_grad') { let G = 0; for (let t = T - 1; t >= 0; t--) { G += traj[t].r; sgd(traj[t].X[traj[t].j], G); } }
      else if (kind === 'sarsa_lin_n') {
        for (let t = 0; t < T; t++) { let G = 0; for (let i = t; i < Math.min(t + hp.n, T); i++) G += traj[i].r; if (t + hp.n < T) G += v(traj[t + hp.n].X[traj[t + hp.n].j]); sgd(traj[t].X[traj[t].j], G); }
      } else if (kind === 'true_online') {
        let z = new Array(D + 1).fill(0), qOld = 0;
        for (let t = 0; t < T; t++) {
          const x = traj[t].X[traj[t].j].concat(1), q = v(x.slice(0, D)), q1 = t + 1 < T ? v(traj[t + 1].X[traj[t + 1].j]) : 0;
          const delta = traj[t].r + q1 - q, zx = dotv(z, x);
          z = z.map((zi, i) => hp.lambda * zi + (1 - hp.alpha * hp.lambda * zx) * x[i]);
          for (let i = 0; i < D; i++) w[i] += hp.alpha * (delta + q - qOld) * z[i] - hp.alpha * (q - qOld) * x[i];
          b += hp.alpha * (delta + q - qOld) * z[D] - hp.alpha * (q - qOld);
          qOld = q1;
        }
      } else if (kind === 'lstd') {
        for (let t = 0; t < T; t++) {
          const x = traj[t].X[traj[t].j].concat(1), x1 = t + 1 < T ? traj[t + 1].X[traj[t + 1].j].concat(1) : new Array(D + 1).fill(0);
          for (let i = 0; i <= D; i++) { if (!x[i]) continue; for (let k = 0; k <= D; k++) A[i][k] += x[i] * (x[k] - x1[k]); bv[i] += traj[t].r * x[i]; }
        }
        if (++episodes % 100 === 0) solve();
      }
    },
    view() { return { kind: 'weights', w: w.slice() }; },
  };
}

// ---------------- gradiente de política lineal ----------------
// Política softmax con preferencias h(x) = θ·x por jugada; crítico / línea de base sobre el estado global.
const glob = X => X[0].slice(0, GD).concat(1);
function polGrad(X, p, j) { const g = new Array(D).fill(0); X.forEach((x, i) => { for (let m = 0; m < D; m++) g[m] -= p[i] * x[m]; }); for (let m = 0; m < D; m++) g[m] += X[j][m]; return g; }
function Reinforce(hp, rng) {
  const th = new Array(D).fill(0), u = new Array(GD + 1).fill(0);
  const probs = X => softmax(X.map(x => dotv(x, th)));
  return {
    type: 'pesos',
    act(ep) { const X = ep.feats(), p = probs(X), j = sample(p, rng); return { X, p, j, why: `sorteo con probabilidad ${(p[j] * 100).toFixed(0)}%` }; },
    greedyIndex(ep) { return argmax(ep.feats().map(x => dotv(x, th))); },
    update() { return null; },
    endEpisode(traj) {
      let G = 0;
      for (let i = traj.length - 1; i >= 0; i--) {
        const { X, j, r } = traj[i];
        G += r;
        const g = glob(X), delta = hp.baseline ? G - dotv(u, g) : G;
        if (hp.baseline) for (let m = 0; m <= GD; m++) u[m] += hp.alphaW * delta * g[m];
        const gr = polGrad(X, probs(X), j);
        for (let m = 0; m < D; m++) th[m] += hp.alpha * delta * gr[m];
      }
    },
    view() { return { kind: 'weights', w: th.slice() }; },
  };
}

function ActorCritic(hp, rng, traces) {
  let th = new Array(D).fill(0), u = new Array(GD + 1).fill(0), zt = null, zu = null;
  const probs = X => softmax(X.map(x => dotv(x, th)));
  return {
    type: 'pesos',
    act(ep) { const X = ep.feats(), p = probs(X), j = sample(p, rng); return { X, p, j, why: `sorteo con probabilidad ${(p[j] * 100).toFixed(0)}%` }; },
    greedyIndex(ep) { return argmax(ep.feats().map(x => dotv(x, th))); },
    update(step, next) {
      if (!zt) { zt = new Array(D).fill(0); zu = new Array(GD + 1).fill(0); }
      const gs = glob(step.X), V = dotv(u, gs), V1 = next ? dotv(u, glob(next.X)) : 0, delta = step.r + V1 - V;
      const grad = polGrad(step.X, probs(step.X), step.j);
      if (traces) { zu = zu.map((z, i) => hp.lambda * z + gs[i]); zt = zt.map((z, i) => hp.lambda * z + grad[i]); }
      const eu = traces ? zu : gs, et = traces ? zt : grad;
      u = u.map((ui, i) => ui + hp.alphaW * delta * eu[i]);
      th = th.map((t, i) => t + hp.alpha * delta * et[i]);
      if (!next) { zt = null; zu = null; }
      return { text: `crítico: esperaba ${V.toFixed(2)}, ahora ${(step.r + V1).toFixed(2)}  (δ = ${delta.toFixed(3)})\nel actor ${delta >= 0 ? 'sube' : 'baja'} la probabilidad de esta jugada${traces ? ' y, más suave, la de las anteriores (trazas)' : ''}` };
    },
    endEpisode() { zt = null; zu = null; },
    view() { return { kind: 'weights', w: th.slice() }; },
  };
}

// ---------------- TD-Gammon: red chica + TD(λ) sobre los pesos ----------------
function TDGammon(hp, rng) {
  const Hn = 24;
  const W1 = Array.from({ length: Hn }, () => Array.from({ length: D }, () => (rng() - 0.5) * 0.3)), b1 = new Array(Hn).fill(0);
  const W2 = Array.from({ length: Hn }, () => (rng() - 0.5) * 0.1);
  let b2 = 0;
  const fwd = x => { const h = W1.map((r, i) => Math.max(0, dotv(r, x) + b1[i])); return { h, v: dotv(h, W2) + b2 }; };
  let z = null;
  const grad = (x, f) => ({ gW1: W1.map((_, i) => f.h[i] > 0 ? x.map(xj => W2[i] * xj) : new Array(D).fill(0)), gb1: W1.map((_, i) => f.h[i] > 0 ? W2[i] : 0), gW2: f.h.slice() });
  function td(delta) {
    for (let i = 0; i < Hn; i++) { for (let j = 0; j < D; j++) W1[i][j] += hp.alpha * delta * z.gW1[i][j]; b1[i] += hp.alpha * delta * z.gb1[i]; W2[i] += hp.alpha * delta * z.gW2[i]; }
    b2 += hp.alpha * delta * z.gb2;
  }
  function decay(gr) {
    if (!z) { z = { ...gr, gb2: 1 }; return; }
    for (let i = 0; i < Hn; i++) { for (let j = 0; j < D; j++) z.gW1[i][j] = hp.lambda * z.gW1[i][j] + gr.gW1[i][j]; z.gb1[i] = hp.lambda * z.gb1[i] + gr.gb1[i]; z.gW2[i] = hp.lambda * z.gW2[i] + gr.gW2[i]; }
    z.gb2 = hp.lambda * z.gb2 + 1;
  }
  return {
    type: 'pesos',
    act(ep) {
      const X = ep.feats(), F = X.map(fwd), means = F.map(f => f.v);
      if (rng() < hp.epsilon) return { X, F, means, j: rng.int(X.length), why: `explora (ε = ${hp.epsilon})` };
      return { X, F, means, j: argmax(means), why: 'la de mayor valor según la red' };
    },
    greedyIndex(ep) { return argmax(ep.feats().map(x => fwd(x).v)); },
    update(step, next) {
      // TD(λ): el error de este turno corrige la red en la dirección de las trazas de todos los anteriores
      const x = step.X[step.j], f = step.F[step.j];
      decay(grad(x, f));
      const vNext = next ? next.F[next.j].v : 0, delta = step.r + vNext - f.v;
      td(delta);
      if (!next) z = null;
      return { text: `la red esperaba ${f.v.toFixed(2)} hasta el final; con lo que pasó, ${(step.r + vNext).toFixed(2)}  (δ = ${delta.toFixed(3)})\nse corrigen los ${Hn * (D + 2) + 1} pesos, también hacia las jugadas anteriores (trazas λ = ${hp.lambda})` };
    },
    endEpisode() { z = null; },
    view() {
      // importancia de cada característica de entrada: cuánto pesa en la red (Σ |W1| · |W2|)
      const imp = new Array(D).fill(0);
      for (let i = 0; i < Hn; i++) for (let j = 0; j < D; j++) imp[j] += Math.abs(W1[i][j]) * Math.abs(W2[i]);
      return { kind: 'weights', w: imp, unit: 'importancia' };
    },
  };
}

// ---------------- PPO liviano: política lineal, tandas y objetivo recortado ----------------
function PPOLite(hp, rng) {
  let th = new Array(D).fill(0), u = new Array(GD + 1).fill(0), batch = [], lastClip = 0;
  const probs = X => softmax(X.map(x => dotv(x, th)));
  function learn() {
    // ventajas con GAE (γ = 1) por partida
    const data = [];
    for (const traj of batch) {
      let adv = 0;
      for (let t = traj.length - 1; t >= 0; t--) {
        const V = dotv(u, glob(traj[t].X)), V1 = t + 1 < traj.length ? dotv(u, glob(traj[t + 1].X)) : 0, delta = traj[t].r + V1 - V;
        adv = delta + hp.lambda * adv;
        data.push({ X: traj[t].X, j: traj[t].j, pOld: traj[t].p[traj[t].j], A: adv, ret: adv + V });
      }
    }
    const mean = sum(data.map(d => d.A)) / data.length, sd = Math.sqrt(sum(data.map(d => (d.A - mean) ** 2)) / data.length) + 1e-8;
    let clipped = 0, total = 0;
    for (let ep = 0; ep < hp.epochs; ep++) for (const d of data) {
      const A = (d.A - mean) / sd, p = probs(d.X), r = p[d.j] / d.pOld;
      total++;
      if ((A > 0 && r > 1 + hp.clip) || (A < 0 && r < 1 - hp.clip)) { clipped++; continue; } // el freno: sin gradiente
      const gr = polGrad(d.X, p, d.j);
      for (let m = 0; m < D; m++) th[m] += hp.alpha * A * r * gr[m];
      const gs = glob(d.X), err = d.ret - dotv(u, gs);
      for (let i = 0; i <= GD; i++) u[i] += hp.alphaW * err * gs[i];
    }
    lastClip = total ? clipped / total : 0;
    batch = [];
  }
  return {
    type: 'pesos',
    act(ep) { const X = ep.feats(), p = probs(X), j = sample(p, rng); return { X, p, j, why: `sorteo con probabilidad ${(p[j] * 100).toFixed(0)}%` }; },
    greedyIndex(ep) { return argmax(ep.feats().map(x => dotv(x, th))); },
    update() { return null; },
    endEpisode(traj) { batch.push(traj); if (batch.length >= hp.batch) learn(); },
    view() { return { kind: 'weights', w: th.slice(), note: `En la última tanda, el freno frenó el ${(lastClip * 100).toFixed(0)}% de las actualizaciones.` }; },
  };
}

// ---------------- neuroevolución: ES y algoritmo genético ----------------
// Política: puntaje lineal por jugada sobre las 65 características (θ de 65).
const chooserOf = th => ep => argmax(ep.feats().map(x => dotv(x, th)));
function fitness(th, seeds) {
  const ch = chooserOf(th);
  let tot = 0;
  for (const s of seeds) { const ep = trainEpisode(E.makeRng(s)); while (!ep.done) ep.step(ch(ep)); tot += ep.result(); }
  return tot / seeds.length;
}
const gauss = rng => { let u = 0, v = 0; while (u === 0) u = rng(); while (v === 0) v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

function ES(hp, rng) {
  const th = new Array(D).fill(0).map(() => gauss(rng) * 0.1), m = new Array(D).fill(0), v = new Array(D).fill(0);
  let gen = 0;
  return {
    type: 'pob',
    generation() {
      gen++;
      const seeds = Array.from({ length: hp.rounds }, () => rng.int(2 ** 31));
      const pairs = hp.pop / 2, eps = Array.from({ length: pairs }, () => th.map(() => gauss(rng)));
      const kids = [...eps.map(e => th.map((x, i) => x + hp.sigma * e[i])), ...eps.map(e => th.map((x, i) => x - hp.sigma * e[i]))];
      const fit = kids.map(k => fitness(k, seeds));
      const rank = new Array(fit.length);
      fit.map((f, i) => [f, i]).sort((a, b) => a[0] - b[0]).forEach(([, i], r) => rank[i] = r / (fit.length - 1) - 0.5);
      const grad = th.map((_, i) => eps.reduce((s, e, p) => s + (rank[p] - rank[p + pairs]) * e[i], 0) / (pairs * hp.sigma));
      for (let i = 0; i < D; i++) {
        m[i] = 0.9 * m[i] + 0.1 * grad[i]; v[i] = 0.999 * v[i] + 0.001 * grad[i] ** 2;
        th[i] += hp.alpha * (m[i] / (1 - 0.9 ** gen)) / (Math.sqrt(v[i] / (1 - 0.999 ** gen)) + 1e-8);
      }
      return { gen, fit, survivors: [], parentFit: fitness(th, seeds) };
    },
    greedyIndex(ep) { return chooserOf(th)(ep); },
    view() { return { kind: 'weights', w: th.slice() }; },
  };
}

function GA(hp, rng) {
  let pop = Array.from({ length: hp.pop }, (_, i) => ({ th: new Array(D).fill(0).map(() => gauss(rng) * 0.3), id: i, parents: [] }));
  let gen = 0, nextId = hp.pop, best = pop[0];
  return {
    type: 'pob',
    generation() {
      gen++;
      const seeds = Array.from({ length: hp.rounds }, () => rng.int(2 ** 31));
      pop.forEach(ind => ind.f = fitness(ind.th, seeds));
      pop.sort((a, b) => b.f - a.f);
      best = pop[0];
      const fit = pop.map(p => p.f), elite = pop.slice(0, hp.elite);
      const pickParent = () => { const a = pop[rng.int(pop.length)], b = pop[rng.int(pop.length)]; return a.f > b.f ? a : b; }; // torneo de 2
      const children = [];
      while (children.length < hp.pop - hp.elite) {
        const p1 = pickParent(), p2 = pickParent();
        const th = p1.th.map((x, i) => (rng() < 0.5 ? x : p2.th[i]) + (rng() < hp.mutRate ? gauss(rng) * hp.sigma : 0));
        children.push({ th, id: nextId++, parents: [p1.id, p2.id] });
      }
      pop = elite.map(e => ({ th: e.th, id: e.id, parents: e.parents })).concat(children);
      return { gen, fit, survivors: fit.slice(0, hp.elite), parentFit: best.f, lineage: `el mejor es #${best.id}${best.parents.length ? `, hijo de #${best.parents[0]} y #${best.parents[1]}` : ' (de la población inicial)'}` };
    },
    greedyIndex(ep) { return chooserOf(best.th)(ep); },
    view() { return { kind: 'weights', w: best.th.slice() }; },
  };
}

// ---------------- búsqueda (no aprende: piensa en cada jugada) ----------------
// ISMCTS de una jugada: UCB entre las candidatas, manos posibles para los rivales
// (que juegan como Afinado o, si se da, con `oppAgent`), una vuelta de mesa y la
// evaluación del Experto. Devuelve el estado de la raíz en cada punto de control.
function scoreAfter(sim, me) {
  let opp = -Infinity;
  for (let q = 0; q < sim.n; q++) if (q !== me) opp = Math.max(opp, E.evaluate(sim, q));
  return (E.evaluate(sim, me) - E.EXPERTO_W.opp * opp) / E.EXPERTO_W.win;
}
function rollout(sim, me, a, rng, oppAgent) {
  sim.step(a);
  const stop = sim.turn + sim.n;
  while (!sim.done && sim.turn < stop) sim.step((sim.cur !== me && oppAgent ? oppAgent : botOf('afinado')).act(sim, rng));
  return scoreAfter(sim, me);
}
function searchRoot(g, iterations, rng, c = 0.7, checkpoints = null, top = 8, oppAgent = null) {
  const me = g.cur, cands = E.ranked(g, g.legalActions(), E.AFINADO).slice(0, top);
  const n = new Array(cands.length).fill(0), w = new Array(cands.length).fill(0), snaps = [];
  for (let it = 1; it <= iterations; it++) {
    let j = n.indexOf(0);
    if (j < 0) { let bu = -Infinity; n.forEach((ni, i) => { const u = w[i] / ni + c * Math.sqrt(Math.log(it) / ni); if (u > bu) { bu = u; j = i; } }); }
    w[j] += rollout(E.determinize(g, me, rng), me, cands[j], rng, oppAgent); n[j]++;
    if (checkpoints && checkpoints.includes(it)) snaps.push({ it, cands: cands.map((a, i) => [a, n[i], n[i] ? w[i] / n[i] : null]) });
  }
  return { best: cands[argmax(n)], cands, n, w, snaps };
}
// Experto: misma cantidad de manos imaginadas para cada candidata.
function expertRoot(g, samples, rng, top = 6) {
  const me = g.cur, cands = E.ranked(g, g.legalActions(), E.AFINADO).slice(0, top);
  const means = cands.map(a => { let s = 0; for (let i = 0; i < samples; i++) s += rollout(E.determinize(g, me, rng), me, a, rng); return s / samples; });
  return { best: cands[argmax(means)], cands, means };
}
const BUDGETS = [10, 30, 100, 300, 1000];
function randomSituation(rng) {
  const ep = new Episode(['afinado', 'coleccionista', 'goloso'], rng), skip = rng.int(4), af = botOf('afinado');
  for (let t = 0; t < skip && !ep.done; t++) ep.step(E.ranked(ep.g, ep.legal(), E.AFINADO).length ? ep.legal().indexOf(af.act(ep.g, rng)) : 0);
  return ep;
}
const snapState = g => ({ rows: g.rows.map(r => r.slice()), hand: g.hands[g.cur].slice(), col: g.cols[g.cur].slice(), nOpp: g.n - 1 });
function Planner(hp, rng) {
  let view = null;
  const idxOf = (ep, a) => ep.legal().indexOf(a);
  return {
    type: 'plan',
    act(ep) {
      const s = searchRoot(ep.g, hp.iterations, rng, hp.c, [hp.iterations]);
      const k = s.cands.indexOf(s.best);
      return { j: idxOf(ep, s.best), cands: s.cands, means: s.snaps[0].cands.map(c => c[2]), visits: s.n, why: `la más simulada (${s.n[k]} de ${hp.iterations})` };
    },
    greedyIndex(ep) { return idxOf(ep, searchRoot(ep.g, hp.iterations, rng, hp.c).best); },
    chooserFor(iters) { return ep => idxOf(ep, searchRoot(ep.g, iters, rng, hp.c).best); },
    update() { return null },
    endEpisode() {},
    think() {
      const ep = randomSituation(rng);
      if (ep.done) return this.think();
      const cps = [10, 25, 50, 100, 200, 500, 1000, 2000].filter(x => x <= hp.iterations);
      if (!cps.includes(hp.iterations)) cps.push(hp.iterations);
      const s = searchRoot(ep.g, hp.iterations, rng, hp.c, cps);
      view = { kind: 'tree', ...snapState(ep.g), best: s.best, snaps: s.snaps, iterations: hp.iterations };
    },
    view() { if (!view) this.think(); return view; },
  };
}
function Expert(hp, rng) {
  let view = null;
  const idxOf = (ep, a) => ep.legal().indexOf(a);
  return {
    type: 'plan', budgets: [1, 3, 8, 20, 60], unit: 'manos imaginadas',
    act(ep) {
      const s = expertRoot(ep.g, hp.samples, rng);
      return { j: idxOf(ep, s.best), cands: s.cands, means: s.means, why: `la mejor evaluada en ${hp.samples} manos imaginadas por jugada` };
    },
    greedyIndex(ep) { return idxOf(ep, expertRoot(ep.g, hp.samples, rng).best); },
    chooserFor(samples) { return ep => idxOf(ep, expertRoot(ep.g, samples, rng).best); },
    update() { return null; },
    endEpisode() {},
    think() {
      const ep = randomSituation(rng);
      if (ep.done) return this.think();
      const cps = [1, 3, 10, 30].filter(x => x < hp.samples).concat(hp.samples);
      // cada punto es una búsqueda nueva con más manos imaginadas
      const snaps = cps.map(it => { const s = expertRoot(ep.g, it, rng); return { it, cands: s.cands.map((a, i) => [a, it, s.means[i]]) }; });
      const last = snaps[snaps.length - 1].cands;
      view = { kind: 'tree', mode: 'samples', ...snapState(ep.g), best: last.reduce((b, c) => c[2] > b[2] ? c : b)[0], snaps, iterations: hp.samples };
    },
    view() { if (!view) this.think(); return view; },
  };
}

// ---------------- Expert Iteration liviano ----------------
// Experto: búsqueda que imagina a los rivales con el aprendiz. Aprendiz: política
// lineal que imita cómo reparte el experto sus simulaciones entre las candidatas.
function ExItLite(hp, rng) {
  const th = new Array(D).fill(0), agree = [];
  const probs = X => softmax(X.map(x => dotv(x, th)));
  const apprentice = { act(g, r) { if (g.phase !== 'play') return botOf('afinado').act(g, r); const [legal, X] = E.decision(g); return legal[argmax(X.map(x => dotv(Array.from(x), th)))]; } };
  return {
    type: 'pesos', curveEvery: 10, evalEvery: 60,
    act(ep) {
      const X = ep.feats(), p = probs(X), legal = ep.legal();
      const s = searchRoot(ep.g, hp.iterations, rng, 0.7, null, 8, apprentice);
      const tot = sum(s.n), target = legal.map(a => { const k = s.cands.indexOf(a); return k < 0 ? 0 : s.n[k] / tot; });
      const j = legal.indexOf(s.best);
      return { X, p, target, j, cands: s.cands, visits: s.n, why: `el experto la simuló más (${s.n[s.cands.indexOf(s.best)]} de ${tot}); el aprendiz le daba ${(p[j] * 100).toFixed(0)}%` };
    },
    greedyIndex(ep) { return argmax(ep.feats().map(x => dotv(x, th))); },
    update(step) {
      // el aprendiz se corrige para parecerse al experto: ∇ entropía cruzada = (objetivo − π) · x
      const p = probs(step.X);
      agree.push(argmax(p) === step.j ? 1 : 0); if (agree.length > 300) agree.shift();
      step.X.forEach((x, i) => { const d = step.target[i] - p[i]; if (d) for (let m = 0; m < D; m++) th[m] += hp.alpha * d * x[m]; });
      const p2 = probs(step.X);
      return { text: `el experto le dio a esta jugada el ${(step.target[step.j] * 100).toFixed(0)}% de sus simulaciones; el aprendiz pasa de ${(p[step.j] * 100).toFixed(1)}% a ${(p2[step.j] * 100).toFixed(1)}%` };
    },
    endEpisode() {},
    view() { return { kind: 'weights', w: th.slice(), note: agree.length ? `En las últimas ${agree.length} jugadas, el aprendiz eligió lo mismo que el experto el ${(100 * sum(agree) / agree.length).toFixed(0)}% de las veces.` : '' }; },
  };
}

// ---------------- catálogo ----------------
const ALGS = {
  bandido: { make: (hp, r) => Tabular(hp, r, 'bandido'), hp: { epsilon: 0.1 } },
  ucb: { make: (hp, r) => Tabular(hp, r, 'ucb'), hp: { c: 0.2 } },
  optimista: { make: (hp, r) => Tabular(hp, r, 'optimista'), hp: { epsilon: 0, alpha: 0.1, init: 0.5 } },
  grad_bandit: { make: (hp, r) => GradBandit(hp, r), hp: { alpha: 0.1 } },
  mc: { make: (hp, r) => Tabular(hp, r, 'mc'), hp: { epsilon: 0.1 } },
  mc_off: { make: (hp, r) => Tabular(hp, r, 'mc_off'), hp: { epsilon: 0.2 } },
  sarsa: { make: (hp, r) => Tabular(hp, r, 'sarsa'), hp: { epsilon: 0.1, alpha: 0.05 } },
  q_learning: { make: (hp, r) => Tabular(hp, r, 'q_learning'), hp: { epsilon: 0.1, alpha: 0.05 } },
  expected_sarsa: { make: (hp, r) => Tabular(hp, r, 'expected_sarsa'), hp: { epsilon: 0.1, alpha: 0.05 } },
  double_q: { make: (hp, r) => Tabular(hp, r, 'double_q'), hp: { epsilon: 0.1, alpha: 0.05 } },
  sarsa_n: { make: (hp, r) => Tabular(hp, r, 'sarsa_n'), hp: { epsilon: 0.1, alpha: 0.05, n: 3 } },
  tree_backup: { make: (hp, r) => Tabular(hp, r, 'tree_backup'), hp: { epsilon: 0.2, alpha: 0.05, n: 3 } },
  sarsa_lambda: { make: (hp, r) => Tabular(hp, r, 'sarsa_lambda'), hp: { epsilon: 0.1, alpha: 0.05, lambda: 0.8 } },
  q_lambda: { make: (hp, r) => Tabular(hp, r, 'q_lambda'), hp: { epsilon: 0.1, alpha: 0.05, lambda: 0.8 } },
  dyna_q: { make: (hp, r) => Tabular(hp, r, 'dyna_q'), hp: { epsilon: 0.1, alpha: 0.05, planning: 5 } },
  prioritized: { make: (hp, r) => Tabular(hp, r, 'prioritized'), hp: { epsilon: 0.1, alpha: 0.05, planning: 5, theta: 0.01 } },
  bandido_lineal: { make: (hp, r) => Linear(hp, r, 'bandido_lineal'), hp: { epsilon: 0.1, alpha: 0.01 } },
  mc_grad: { make: (hp, r) => Linear(hp, r, 'mc_grad'), hp: { epsilon: 0.1, alpha: 0.01 } },
  sarsa_lin: { make: (hp, r) => Linear(hp, r, 'sarsa_lin'), hp: { epsilon: 0.1, alpha: 0.01 } },
  sarsa_lin_n: { make: (hp, r) => Linear(hp, r, 'sarsa_lin_n'), hp: { epsilon: 0.1, alpha: 0.01, n: 3 } },
  q_lin: { make: (hp, r) => Linear(hp, r, 'q_lin'), hp: { epsilon: 0.1, alpha: 0.01 } },
  true_online: { make: (hp, r) => Linear(hp, r, 'true_online'), hp: { epsilon: 0.1, alpha: 0.01, lambda: 0.8 } },
  lstd: { make: (hp, r) => Linear(hp, r, 'lstd'), hp: { epsilon: 0.1 } },
  reinforce: { make: (hp, r) => Reinforce(hp, r), hp: { alpha: 0.05, baseline: 1, alphaW: 0.02 } },
  actor_critic: { make: (hp, r) => ActorCritic(hp, r, false), hp: { alpha: 0.05, alphaW: 0.02 } },
  actor_critic_lambda: { make: (hp, r) => ActorCritic(hp, r, true), hp: { alpha: 0.05, alphaW: 0.02, lambda: 0.8 } },
  ppo: { make: (hp, r) => PPOLite(hp, r), hp: { alpha: 0.02, alphaW: 0.02, lambda: 0.95, batch: 20, epochs: 4, clip: 0.2 } },
  td_gammon: { make: (hp, r) => TDGammon(hp, r), hp: { epsilon: 0.05, alpha: 0.005, lambda: 0.7 } },
  es: { make: (hp, r) => ES(hp, r), hp: { pop: 20, rounds: 15, sigma: 0.1, alpha: 0.05 } },
  ga: { make: (hp, r) => GA(hp, r), hp: { pop: 20, rounds: 15, elite: 4, sigma: 0.2, mutRate: 0.3 } },
  experto: { make: (hp, r) => Expert(hp, r), hp: { samples: 8 } },
  ismcts: { make: (hp, r) => Planner(hp, r), hp: { iterations: 100, c: 0.7 } },
  exit: { make: (hp, r) => ExItLite(hp, r), hp: { iterations: 40, alpha: 0.05 } },
};

// ---------------- bucle ----------------
let alg = null, algName = null, rng = null, running = false, rounds = 0, gens = 0;
let curve = [], evals = [], recent = [], popHist = [];
let refs = null;

function references() {
  if (refs) return refs;
  const az = botOf('azar'), af = botOf('afinado');
  refs = {
    azar: evaluate(ep => ep.legal().indexOf(az.act(ep.g, ep.g.rng))),
    afinado: evaluate(ep => ep.legal().indexOf(af.act(ep.g, ep.g.rng))),
  };
  return refs;
}

function playTrainingEpisode(record) {
  const ep = trainEpisode(rng, alg.shaping !== false), traj = [];
  while (!ep.done) {
    const act = alg.act(ep);
    const before = record ? { ...snapState(ep.g), legal: ep.legal() } : null;
    const r = ep.step(act.j);
    const step = { ...act, r };
    if (traj.length) {
      const info = alg.update(traj[traj.length - 1], step, traj);
      if (record && record.length && info) record[record.length - 1].update = info;
    }
    traj.push(step);
    if (record) {
      // candidatas para mostrar: las que dio la búsqueda, o las 6 mejores según lo que estima (más la elegida)
      const scores = act.means || (act.ks && alg.q ? act.ks.map(alg.q) : act.p);
      let show;
      if (act.cands) show = act.cands.map((a, i) => ({ a, v: act.means ? act.means[i] : null, n: act.visits ? act.visits[i] : null }));
      else {
        const idx = before.legal.map((_, i) => i).sort((x, y) => (scores ? scores[y] - scores[x] : 0)).slice(0, 6);
        if (!idx.includes(act.j)) idx.push(act.j);
        show = idx.map(i => ({ a: before.legal[i], v: act.means ? act.means[i] : act.ks && alg.q ? alg.q(act.ks[i]) : null, p: act.p ? act.p[i] : null }));
      }
      record.push({ ...before, legal: undefined, show, a: before.legal[act.j], k: act.ks ? act.ks[act.j] : null, why: act.why, r });
    }
  }
  const info = traj.length ? alg.update(traj[traj.length - 1], null, traj) : null;
  if (record && info) record[record.length - 1].update = info;
  if (traj.length) alg.endEpisode(traj);
  rounds++;
  return { result: ep.result(), n: ep.n, won: ep.g.winners.includes(ep.me), turns: ep.g.turn };
}

function post(extra = {}) {
  self.postMessage({ type: 'state', alg: algName, algType: alg.type, budgets: alg.budgets || BUDGETS, rounds, gens, curve, evals, refs, view: alg.view(), popHist, running, ...extra });
}

const greedyChooser = () => ep => alg.greedyIndex(ep);
function tick() {
  if (!running) return;
  const t0 = Date.now();
  if (alg.type === 'plan') {
    const budgets = alg.budgets || BUDGETS, next = budgets.find(b => !evals.some(e => e[0] === b));
    if (next === undefined) { running = false; post(); return; }
    evals.push([next, evaluate(alg.chooserFor(next), 30)]);
    evals.sort((a, b) => a[0] - b[0]);
    post({ measuring: budgets.find(b => !evals.some(e => e[0] === b)) || null });
    setTimeout(tick, 0);
    return;
  }
  if (alg.type === 'pob') {
    const r = alg.generation();
    gens = r.gen;
    popHist.push({ gen: r.gen, fit: r.fit, survivors: r.survivors, parentFit: r.parentFit, lineage: r.lineage });
    if (popHist.length > 60) popHist.shift();
    if (r.gen % 2 === 0) evals.push([r.gen, evaluate(greedyChooser(), 40)]);
    post();
    setTimeout(tick, 0);
    return;
  }
  const every = alg.curveEvery || 100, evalEvery = alg.evalEvery || 1000;
  while (Date.now() - t0 < 150) {
    recent.push(playTrainingEpisode(null).result);
    if (recent.length >= every) {
      curve.push([rounds, sum(recent) / recent.length]);
      recent = [];
      if (curve.length > 400) curve = curve.filter((_, i) => i % 2 === 0);
    }
    if (rounds % evalEvery === 0) evals.push([rounds, evaluate(greedyChooser())]);
  }
  post();
  setTimeout(tick, 0);
}

self.onmessage = e => {
  const m = e.data;
  if (m.type === 'reset') {
    running = false;
    algName = m.alg;
    rng = E.makeRng(m.seed || 1);
    alg = ALGS[m.alg].make(Object.assign({}, ALGS[m.alg].hp, m.hp || {}), rng);
    rounds = 0; gens = 0; curve = []; evals = []; recent = []; popHist = [];
    references();
    if (alg.type !== 'plan') evals.push([0, evaluate(greedyChooser(), alg.type === 'pob' ? 40 : 120)]);
    post();
  } else if (m.type === 'run') { if (!running) { running = true; tick(); } }
  else if (m.type === 'pause') { running = false; post(); }
  else if (m.type === 'think') { if (alg.think) { alg.think(); post(); } }
  else if (m.type === 'slow') {
    // una partida entera grabada jugada por jugada, aprendiendo
    if (alg.type === 'pob') return;
    const rec = [];
    const res = playTrainingEpisode(rec);
    post({ slow: { turns: rec, ...res } });
  }
};
self.postMessage({ type: 'ready', defaults: Object.fromEntries(Object.entries(ALGS).map(([k, v]) => [k, v.hp])) });
