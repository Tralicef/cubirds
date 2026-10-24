/*
 * Motor de CuBirds para el navegador: reglas, bots, agentes entrenados y
 * torneos con Elo. Replica cubirds/game/engine.py (reglas), cubirds/game/bots.py
 * (bots heurísticos) y cubirds/arena (Elo y torneos): con la misma semilla una
 * partida es idéntica en Python y acá (tests/test_js_parity.py lo comprueba).
 *
 * Se usa como script normal (expone window.CuBirdsEngine), con require() en
 * Node, o como Web Worker (new Worker('engine.js')): recibe {type:'run', ...}
 * y responde con {type:'progress'|'done'|'error'}.
 */
(function (root) {
  'use strict';

  // ---------- reglas ----------
  const SPECIES = ['Flamenco', 'Búho', 'Tucán', 'Pato', 'Loro', 'Urraca', 'Carricero', 'Petirrojo'];
  const COUNTS = [7, 10, 10, 13, 13, 17, 20, 20];
  const SMALL = [2, 3, 3, 4, 4, 5, 6, 6];
  const BIG = [3, 4, 4, 6, 6, 7, 9, 9];
  const NS = 8, NROWS = 4, HAND = 8;
  const A_NODRAW = 64, A_DRAW = 65, A_FLOCK0 = 66, A_NOFLOCK = 74, NACTIONS = 75;
  const MAX_TURNS = 1000;
  const MAX_TABLE = 5;

  const playAction = (s, r, side) => s * 8 + r * 2 + side;
  const decodePlay = a => [a >> 3, (a & 7) >> 1, a & 1];

  // RNG reproducible (mulberry32), igual que Rng en engine.py
  function makeRng(seed) {
    let a = (seed >>> 0) || 1;
    const next = () => {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
    next.int = n => Math.floor(next() * n);
    next.pick = arr => arr[next.int(arr.length)];
    next.shuffle = arr => { for (let i = arr.length - 1; i > 0; i--) { const j = next.int(i + 1); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; };
    next.state = () => a >>> 0;
    return next;
  }

  const hasWon = col => col.filter(c => c > 0).length >= 7 || col.filter(c => c >= 3).length >= 2;

  // Cuántas cartas encierra jugar la especie s en ese lado de la fila.
  function enclosed(row, s, side) {
    const n = row.length;
    for (let k = 0; k < n; k++) if (row[side === 0 ? k : n - 1 - k] === s) return k;
    return 0;
  }

  class Game {
    constructor(n, seed = 0, rng = null) {
      if (n < 2 || n > MAX_TABLE) throw new Error('CuBirds se juega de 2 a 5');
      this.n = n;
      this.rng = rng || makeRng(seed);
      const deck = [];
      for (let s = 0; s < NS; s++) for (let i = 0; i < COUNTS[s]; i++) deck.push(s);
      this.deck = this.rng.shuffle(deck);
      this.discard = [];
      this.rows = [];
      for (let r = 0; r < NROWS; r++) {
        const row = [];
        while (row.length < 3) {
          const c = this.deck.pop();
          if (row.includes(c)) this.discard.push(c); else row.push(c);
        }
        this.rows.push(row);
      }
      this.hands = Array.from({ length: n }, () => new Array(NS).fill(0));
      for (let p = 0; p < n; p++) for (let i = 0; i < HAND; i++) this.hands[p][this.deck.pop()]++;
      this.cols = Array.from({ length: n }, () => new Array(NS).fill(0));
      for (let p = 0; p < n; p++) this.cols[p][this.deck.pop()]++;
      this.cur = 0;
      this.phase = 'play';
      this.turn = 0;
      this.round = 1;
      this.done = false;
      this.winners = [];
      this.endReason = null;
      this.log = [];
    }

    clone(rng) {
      const g = Object.create(Game.prototype);
      Object.assign(g, this);
      g.rng = rng || this.rng;
      g.deck = this.deck.slice(); g.discard = this.discard.slice();
      g.rows = this.rows.map(r => r.slice());
      g.hands = this.hands.map(h => h.slice());
      g.cols = this.cols.map(c => c.slice());
      g.winners = this.winners.slice();
      g.log = [];
      return g;
    }

    toJSON() {
      const { n, deck, discard, rows, hands, cols, cur, phase, turn, round, done, winners, endReason } = this;
      return { n, deck, discard, rows, hands, cols, cur, phase, turn, round, done, winners, endReason, rng: this.rng.state() };
    }

    static fromJSON(o) {
      const g = Object.create(Game.prototype);
      Object.assign(g, JSON.parse(JSON.stringify(o)));
      g.rng = makeRng(o.rng);
      g.log = [];
      return g;
    }

    _draw() {
      if (!this.deck.length) {
        if (!this.discard.length) return null;
        this.deck = this.rng.shuffle(this.discard);
        this.discard = [];
      }
      return this.deck.pop();
    }

    legalActions() {
      if (this.done) return [];
      const h = this.hands[this.cur];
      if (this.phase === 'play') {
        const out = [];
        for (let s = 0; s < NS; s++) if (h[s]) for (let r = 0; r < NROWS; r++) out.push(playAction(s, r, 0), playAction(s, r, 1));
        return out;
      }
      if (this.phase === 'draw') return [A_NODRAW, A_DRAW];
      const out = [];
      for (let s = 0; s < NS; s++) if (h[s] >= SMALL[s]) out.push(A_FLOCK0 + s);
      out.push(A_NOFLOCK);
      return out;
    }

    step(a) {
      const p = this.cur, h = this.hands[p];
      if (this.phase === 'play') {
        const [s, r, side] = decodePlay(a);
        const k = h[s];
        if (!k) throw new Error('especie que no está en la mano');
        h[s] = 0;
        let row = this.rows[r];
        const e = enclosed(row, s, side);
        const taken = side === 0 ? row.slice(0, e) : row.slice(row.length - e);
        const birds = new Array(k).fill(s);
        row = side === 0 ? birds.concat(row.slice(e)) : row.slice(0, row.length - e).concat(birds);
        for (const c of taken) h[c]++;
        const refill = [];
        if (taken.length && new Set(row).size === 1) {
          while (new Set(row).size === 1) {
            const c = this._draw();
            if (c === null) break;
            refill.push(c);
            row = side === 1 ? row.concat([c]) : [c].concat(row);
          }
        }
        this.rows[r] = row;
        this.log.push({ t: 'play', p, s, k, r, side, taken, refill });
        this.phase = taken.length ? 'flock' : 'draw';
      } else if (this.phase === 'draw') {
        const got = [];
        if (a === A_DRAW) for (let i = 0; i < 2; i++) { const c = this._draw(); if (c !== null) { h[c]++; got.push(c); } }
        this.log.push({ t: 'draw', p, got });
        this.phase = 'flock';
      } else {
        if (a !== A_NOFLOCK) {
          const s = a - A_FLOCK0, k = h[s];
          if (k < SMALL[s]) throw new Error('no alcanza para una bandada');
          const keep = k >= BIG[s] ? 2 : 1;
          h[s] = 0;
          this.cols[p][s] += keep;
          for (let i = 0; i < k - keep; i++) this.discard.push(s);
          this.log.push({ t: 'flock', p, s, k, keep });
        }
        this._endTurn();
      }
    }

    _endTurn() {
      const p = this.cur;
      this.turn++;
      if (hasWon(this.cols[p])) return this._finish([p], 'win');
      if (this.hands[p].every(x => x === 0)) {
        // Fin de la ronda: los demás descartan y el que vació reparte y sigue jugando.
        for (let q = 0; q < this.n; q++) for (let s = 0; s < NS; s++) {
          for (let i = 0; i < this.hands[q][s]; i++) this.discard.push(s);
          this.hands[q][s] = 0;
        }
        if (this.deck.length < HAND * this.n) {
          this.deck = this.rng.shuffle(this.discard).concat(this.deck);
          this.discard = [];
        }
        if (this.deck.length < HAND * this.n) return this._finishByCount('deck');
        for (let i = 0; i < this.n; i++) {
          const q = (p + i) % this.n;
          for (let j = 0; j < HAND; j++) this.hands[q][this.deck.pop()]++;
        }
        this.round++;
        this.log.push({ t: 'round', p });
      } else {
        this.cur = (p + 1) % this.n;
      }
      if (this.turn >= MAX_TURNS) return this._finishByCount('cap');
      this.phase = 'play';
    }

    _finishByCount(why) {
      const tot = this.cols.map(c => c.reduce((a, b) => a + b, 0));
      const m = Math.max(...tot);
      this._finish(tot.map((t, q) => t === m ? q : -1).filter(q => q >= 0), why);
    }

    _finish(winners, why) {
      this.done = true;
      this.winners = winners;
      this.endReason = why;
      this.log.push({ t: 'end', why, winners: winners.slice() });
    }

    // Puesto de cada jugador (0 = ganó); los que no ganaron empatan entre sí.
    ranks() { return Array.from({ length: this.n }, (_, q) => this.winners.includes(q) ? 0 : 1); }
  }

  // ---------- bots heurísticos (cubirds/game/bots.py) ----------
  function progress(col) {
    const dist7 = Math.max(0, 7 - col.filter(c => c > 0).length);
    const need = col.map(c => c < 3 ? Math.floor((4 - c) / 2) : 0).sort((a, b) => a - b);
    return Math.min(dist7, need[0] + need[1]);
  }
  // Pesos de la evaluación de la mano. AFINADO salió de una búsqueda aleatoria de
  // 150 combinaciones contra Coleccionista y Goloso: ganó 50% en mesas de 3 (lo justo es 33%).
  const DEFAULT = { big: 2.2, small_base: 1.0, slope: 0.6, below: 0.7, w_new: 1.5, w_part: 1.2, w_done: 0.6, draw: 0.35, take: 0.05 };
  const AFINADO = { big: 4.31, small_base: 1.37, slope: 0.495, below: 0.289, w_new: 0.946, w_part: 3.279, w_done: 0.41, draw: 0.547, take: 0.086 };
  const speciesWeight = (col, s, P = DEFAULT) => col[s] === 0 ? P.w_new : col[s] < 3 ? P.w_part : P.w_done;
  function handValue(hand, col, P = DEFAULT) {
    let v = 0;
    for (let s = 0; s < NS; s++) {
      const k = hand[s];
      if (!k) continue;
      let base;
      if (k >= BIG[s]) base = P.big;
      else if (k >= SMALL[s]) base = P.small_base + P.slope * (k - SMALL[s]) / (BIG[s] - SMALL[s]);
      else base = P.below * k / SMALL[s];
      v += base * speciesWeight(col, s, P);
    }
    return v;
  }
  const lexLess = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i]; return false; };
  function flockChoice(hand, col, legal) {
    const before = progress(col);
    let best = A_NOFLOCK, bk = null;
    for (const a of legal) {
      if (a === A_NOFLOCK) continue;
      const s = a - A_FLOCK0, keep = hand[s] >= BIG[s] ? 2 : 1;
      const nc = col.slice(); nc[s] += keep;
      const p = progress(nc);
      if (p >= before && keep === 1) continue;
      const key = [p, -keep, hand[s]];
      if (bk === null || lexLess(key, bk)) { best = a; bk = key; }
    }
    return best;
  }
  function takenBy(row, s, side) {
    const e = enclosed(row, s, side);
    return side === 0 ? row.slice(0, e) : row.slice(row.length - e);
  }

  function bot(play, draw, flock) {
    return {
      act(g, rng) {
        const legal = g.legalActions();
        if (g.phase === 'play') return play(g, legal, rng);
        if (g.phase === 'draw') return draw ? draw(g, legal, rng) : A_DRAW;
        return flock ? flock(g, legal, rng) : flockChoice(g.hands[g.cur], g.cols[g.cur], legal);
      },
    };
  }
  function coleccionistaScores(g, legal, P = DEFAULT) {
    const h = g.hands[g.cur], col = g.cols[g.cur];
    const base = handValue(h, col, P);
    return legal.map(a => {
      const [s, r, side] = decodePlay(a);
      const taken = takenBy(g.rows[r], s, side);
      const nh = h.slice(); nh[s] = 0;
      for (const c of taken) nh[c]++;
      return handValue(nh, col, P) - base + (taken.length ? P.take * taken.length : P.draw);
    });
  }
  // Jugadas de mejor a peor según Coleccionista (a igual puntaje, en el orden de legal).
  function ranked(g, legal, P = DEFAULT) {
    const sc = coleccionistaScores(g, legal, P);
    return legal.map((a, i) => i).sort((i, j) => sc[j] - sc[i]).map(i => legal[i]);
  }
  // Copia de la partida con las cartas ocultas (mazo, descarte y manos ajenas) repartidas al azar.
  function determinize(g, me, rng) {
    const sim = g.clone();
    const pool = sim.deck.concat(sim.discard);
    for (let q = 0; q < sim.n; q++) if (q !== me) for (let s = 0; s < NS; s++) for (let i = 0; i < sim.hands[q][s]; i++) pool.push(s);
    rng.shuffle(pool);
    let pos = 0;
    for (let q = 0; q < sim.n; q++) {
      if (q === me) continue;
      const k = sim.hands[q].reduce((a, b) => a + b, 0);
      sim.hands[q] = new Array(NS).fill(0);
      for (const c of pool.slice(pos, pos + k)) sim.hands[q][c]++;
      pos += k;
    }
    const nd = sim.discard.length;
    sim.discard = pool.slice(pos, pos + nd);
    sim.deck = pool.slice(pos + nd);
    return sim;
  }
  // Pesos de la evaluación del Experto, de una búsqueda aleatoria de 25 combinaciones.
  const EXPERTO_W = { prog: 1.0, hand: 0.28, birds: 0.51, opp: 0.35, win: 12.8 };
  // Qué tan bien está p: menos bandadas por formar, mejor mano y más pájaros.
  function evaluate(g, p, w = EXPERTO_W) {
    if (g.done) return g.winners.includes(p) ? w.win : -w.win;
    return -w.prog * progress(g.cols[p]) + w.hand * handValue(g.hands[p], g.cols[p], AFINADO) + w.birds * g.cols[p].reduce((a, b) => a + b, 0);
  }
  // Monte Carlo: para las mejores jugadas según Afinado, reparte manos posibles a los
  // rivales, simula una vuelta de mesa con Afinado y evalúa la posición contra el mejor rival.
  function experto(rollouts = 8, top = 6, horizon = 1) {
    return bot((g, legal, rng) => {
      const me = g.cur, cands = ranked(g, legal, AFINADO).slice(0, top);
      if (cands.length === 1) return cands[0];
      let best = cands[0], bv = null;
      for (const a of cands) {
        let v = 0;
        for (let i = 0; i < rollouts; i++) {
          const sim = determinize(g, me, rng);
          sim.step(a);
          const stop = sim.turn + horizon * sim.n;
          while (!sim.done && sim.turn < stop) sim.step(H.afinado.act(sim, rng));
          let opp = -Infinity;
          for (let q = 0; q < sim.n; q++) if (q !== me) opp = Math.max(opp, evaluate(sim, q));
          v += evaluate(sim, me) - EXPERTO_W.opp * opp;
        }
        if (bv === null || v > bv) { best = a; bv = v; }
      }
      return best;
    }, (g, l, rng) => H.afinado.act(g, rng), (g, l, rng) => H.afinado.act(g, rng));
  }
  // ISMCTS (Cowling et al., 2012): en cada iteración reparte manos posibles a los rivales
  // y baja por un árbol de jugadas propias eligiendo con UCB entre las candidatas de esa
  // mano posible (las mejores según Afinado); cada hijo cuenta cuántas veces estuvo
  // disponible. Los rivales juegan como Afinado. Tras `depth` jugadas propias se termina
  // la vuelta y se evalúa como el Experto. Sólo existe en el motor JS.
  function ismcts(iterations = 120, top = 6, depth = 2, c = 0.7) {
    const node = () => ({ n: 0, w: 0, avail: 0, kids: new Map() });
    return bot((g, legal, rng) => {
      const me = g.cur, root = node();
      const rootCands = ranked(g, legal, AFINADO).slice(0, top);
      if (rootCands.length === 1) return rootCands[0];
      for (let it = 0; it < iterations; it++) {
        const sim = determinize(g, me, rng);
        const path = [];
        let cur = root, mine = 0;
        while (!sim.done) {
          if (sim.cur === me && sim.phase === 'play' && mine < depth) {
            const cands = mine === 0 ? rootCands : ranked(sim, sim.legalActions(), AFINADO).slice(0, top);
            let best = null, bu = -Infinity;
            for (const a of cands) {
              let k = cur.kids.get(a);
              if (!k) { k = node(); cur.kids.set(a, k); }
              k.avail++;
              const u = k.n === 0 ? Infinity : k.w / k.n + c * Math.sqrt(Math.log(k.avail) / k.n);
              if (u > bu) { bu = u; best = a; }
            }
            cur = cur.kids.get(best);
            path.push(cur);
            sim.step(best);
            mine++;
            if (mine === depth) { var stop = sim.turn + sim.n; }
          } else {
            if (mine === depth && sim.turn >= stop) break;
            sim.step(H.afinado.act(sim, rng));
          }
        }
        let opp = -Infinity;
        for (let q = 0; q < sim.n; q++) if (q !== me) opp = Math.max(opp, evaluate(sim, q));
        const v = (evaluate(sim, me) - EXPERTO_W.opp * opp) / EXPERTO_W.win;
        for (const k of path) { k.n++; k.w += v; }
      }
      let best = rootCands[0], bn = -1;
      for (const a of rootCands) { const k = root.kids.get(a); if (k && k.n > bn) { bn = k.n; best = a; } }
      return best;
    }, (g, l, rng) => H.afinado.act(g, rng), (g, l, rng) => H.afinado.act(g, rng));
  }
  function valueBot(P) {
    return bot((g, legal, rng) => {
      const sc = coleccionistaScores(g, legal, P);
      let best = [], bv = null;
      legal.forEach((a, i) => {
        const v = sc[i];
        if (bv === null || v > bv + 1e-9) { best = [a]; bv = v; }
        else if (v > bv - 1e-9) best.push(a);
      });
      return rng.pick(best);
    });
  }
  const H = {
    azar: bot((g, l, rng) => rng.pick(l), (g, l, rng) => rng.pick(l), (g, l, rng) => rng.pick(l)),
    goloso: bot(
      (g, legal, rng) => {
        const h = g.hands[g.cur];
        let best = [], bk = null;
        for (const a of legal) {
          const [s, r, side] = decodePlay(a);
          const key = [enclosed(g.rows[r], s, side), -h[s]];
          if (bk === null || lexLess(bk, key)) { best = [a]; bk = key; }
          else if (!lexLess(key, bk)) best.push(a);
        }
        return rng.pick(best);
      },
      null,
      (g, legal) => {
        const h = g.hands[g.cur], col = g.cols[g.cur];
        const opts = legal.filter(a => a !== A_NOFLOCK);
        if (!opts.length) return A_NOFLOCK;
        let best = null, bk = null;
        for (const a of opts) {
          const s = a - A_FLOCK0;
          const key = [h[s] >= BIG[s] ? 2 : 1, col[s] === 0 ? 1 : 0, -a];
          if (bk === null || lexLess(bk, key)) { best = a; bk = key; }
        }
        return best;
      },
    ),
    coleccionista: valueBot(DEFAULT),
    afinado: valueBot(AFINADO),
  };

  // ---------- características para los agentes entrenados (cubirds/rl/features.py) ----------
  const G_DIM = 33, A_DIM = 32, DIM = G_DIM + A_DIM;
  const sumArr = a => a.reduce((x, y) => x + y, 0);
  function globalFeatures(g, p = g.cur) {
    const h = g.hands[p], col = g.cols[p], x = new Float64Array(G_DIM);
    const opp = []; for (let q = 0; q < g.n; q++) if (q !== p) opp.push(q);
    for (let s = 0; s < NS; s++) {
      x[s] = h[s] / BIG[s];
      x[8 + s] = Math.min(col[s], 3) / 3;
      x[17 + s] = Math.max(...opp.map(q => Math.min(g.cols[q][s], 3))) / 3;
    }
    x[16] = progress(col) / 4;
    x[25] = Math.min(...opp.map(q => progress(g.cols[q]))) / 4;
    x[26] = opp.reduce((a, q) => a + sumArr(g.hands[q]), 0) / opp.length / 10;
    x[27] = sumArr(h) / 10;
    x[28] = g.deck.length / 60;
    x[29] = (g.n - 2) / 3;
    x[30 + (g.phase === 'play' ? 0 : g.phase === 'draw' ? 1 : 2)] = 1;
    return x;
  }
  function actionFeatures(g, a, p = g.cur) {
    const h = g.hands[p], col = g.cols[p], x = new Float64Array(A_DIM);
    if (a < 64) {
      const [s, r, side] = decodePlay(a);
      const row = g.rows[r], k = enclosed(row, s, side);
      const taken = side === 0 ? row.slice(0, k) : row.slice(row.length - k);
      const nh = h.slice(); nh[s] = 0;
      for (const c of taken) nh[c]++;
      x[s] = 1; x[8] = 1; x[9] = h[s] / BIG[s]; x[10] = k / 6;
      for (const c of taken) x[11 + c] += 1 / BIG[c];
      let ns = 0, nb = 0;
      for (let t = 0; t < NS; t++) { if (nh[t] >= SMALL[t]) ns++; if (nh[t] >= BIG[t]) nb++; }
      x[19] = ns / 3; x[20] = nb / 2;
      x[21] = (handValue(nh, col, DEFAULT) - handValue(h, col, DEFAULT)) / 3;
      x[22] = (handValue(nh, col, AFINADO) - handValue(h, col, AFINADO)) / 5;
      x[23] = k === 0 ? 1 : 0;
      x[24] = (row.length - k + h[s]) / 10;
    } else if (a === A_DRAW) x[25] = 1;
    else if (a === A_NODRAW) x[26] = 1;
    else if (a === A_NOFLOCK) x[30] = 1;
    else {
      const s = a - A_FLOCK0, keep = h[s] >= BIG[s] ? 2 : 1;
      const nc = col.slice(); nc[s] += keep;
      x[s] = 1; x[27] = 1; x[28] = keep / 2;
      x[29] = (progress(col) - progress(nc)) / 2;
      x[31] = Math.min(nc[s], 3) / 3;
    }
    return x;
  }
  // [acciones legales, filas de DIM valores] para el jugador de turno.
  function decision(g) {
    const legal = g.legalActions(), gx = globalFeatures(g);
    return [legal, legal.map(a => { const row = new Float64Array(DIM); row.set(gx); row.set(actionFeatures(g, a), G_DIM); return row; })];
  }

  // ---------- redes entrenadas (cubirds/rl/ppo.py) ----------
  // Capas guardadas como [entrada][salida]; puntúa cada acción legal y elige la mejor.
  function dense(x, W, b, act) {
    const out = new Float64Array(b.length);
    for (let j = 0; j < b.length; j++) {
      let z = b[j];
      for (let i = 0; i < x.length; i++) z += x[i] * W[i][j];
      out[j] = act ? Math.tanh(z) : z;
    }
    return out;
  }
  function netLogit(P, x) { return dense(dense(dense(x, P.W1, P.b1, true), P.W2, P.b2, true), P.W3, P.b3, false)[0]; }
  function netValue(P, gx) { return dense(dense(gx, P.V1, P.c1, true), P.V2, P.c2, false)[0]; }
  function netAgent(model) {
    const P = model.params;
    return {
      act(g) {
        const [legal, X] = decision(g);
        let best = 0, bz = -Infinity;
        X.forEach((x, i) => { const z = netLogit(P, x); if (z > bz) { bz = z; best = i; } });
        return legal[best];
      },
    };
  }
  // ---------- Sutton & Barto tabular (cubirds/rl/sb/common.py: key) ----------
  const bucketOf = (x, edges) => { let i = 0; while (i < edges.length && x >= edges[i]) i++; return i; };
  function sbKey(g, a, p = g.cur) {
    const h = g.hands[p], col = g.cols[p];
    const [s, r, side] = decodePlay(a);
    const row = g.rows[r], k = enclosed(row, s, side);
    const taken = side === 0 ? row.slice(0, k) : row.slice(row.length - k);
    const nh = h.slice(); nh[s] = 0;
    for (const c of taken) nh[c]++;
    let ready = 0, gain = 0;
    const before = progress(col);
    for (let t = 0; t < NS; t++) if (nh[t] >= SMALL[t]) {
      const keep = nh[t] >= BIG[t] ? 2 : 1;
      ready = Math.max(ready, keep);
      const nc = col.slice(); nc[t] += keep;
      gain = Math.max(gain, before - progress(nc));
    }
    let opp = Infinity;
    for (let q = 0; q < g.n; q++) if (q !== p) opp = Math.min(opp, progress(g.cols[q]));
    return [bucketOf(k, [1, 2, 3, 5]), ready, Math.min(gain, 2), bucketOf(before, [2, 3, 4]), bucketOf(opp, [2, 3]),
      bucketOf(h[s] / SMALL[s], [0.5, 1.0]), bucketOf(sumArr(nh), [1, 4, 8])].join(',');
  }
  // Elige la jugada de mayor valor en la tabla; robar y bandadas, como Afinado.
  function tablaAgent(model) {
    const V = model.values, def = model.default;
    return bot(g => {
      const legal = g.legalActions();
      let best = 0, bv = -Infinity;
      legal.forEach((a, i) => { const v = V[sbKey(g, a)] ?? def; if (v > bv) { bv = v; best = i; } });
      return legal[best];
    });
  }
  // Lineal (Parte II de S&B): puntaje w·x + b de cada jugada; robar y bandadas, como Afinado.
  function linealAgent(model) {
    const w = model.w, b = model.b;
    return bot(g => {
      const [legal, X] = decision(g);
      let best = 0, bv = -Infinity;
      X.forEach((x, i) => { let v = b; for (let j = 0; j < x.length; j++) v += w[j] * x[j]; if (v > bv) { bv = v; best = i; } });
      return legal[best];
    });
  }
  // Valor (TD-Gammon): red ReLU que valúa el afterstate de cada jugada.
  function valorAgent(model) {
    const { W1, b1, W2, b2 } = model;
    return bot(g => {
      const [legal, X] = decision(g);
      let best = 0, bv = -Infinity;
      X.forEach((x, i) => {
        let v = b2;
        for (let k = 0; k < b1.length; k++) { let z = b1[k]; for (let j = 0; j < x.length; j++) z += x[j] * W1[j][k]; if (z > 0) v += z * W2[k]; }
        if (v > bv) { bv = v; best = i; }
      });
      return legal[best];
    });
  }
  const MODEL_KINDS = ['red', 'tabla', 'lineal', 'valor'];
  // En Node: lee web/arena/models (index.json + archivos) y le pega a cada modelo su ficha.
  function loadModels(dir) {
    const fs = require('fs'), path = require('path');
    const file = path.join(dir, 'index.json');
    if (!fs.existsSync(file)) return {};
    const out = {};
    for (const m of JSON.parse(fs.readFileSync(file))) out[m.id] = Object.assign(JSON.parse(fs.readFileSync(path.join(dir, m.file))), { label: m.label, family: m.family, desc: m.desc });
    return out;
  }

  // ---------- catálogo ----------
  const CATALOG = [
    { spec: 'azar', label: 'Azar', family: 'heurístico', desc: 'Elige cualquier jugada legal al azar.' },
    { spec: 'goloso', label: 'Goloso', family: 'heurístico', desc: 'Se lleva la mayor cantidad de pájaros posible; a igualdad, juega menos cartas. Siempre roba y forma la bandada más grande.' },
    { spec: 'coleccionista', label: 'Coleccionista', family: 'heurístico', desc: 'Puntúa su mano según qué tan cerca está de formar bandadas de especies que le faltan y elige la jugada que más la mejora.' },
    { spec: 'afinado', label: 'Afinado', family: 'heurístico', desc: 'Coleccionista con pesos encontrados por búsqueda aleatoria: valora mucho más completar especies que ya tiene (camino a 2 especies con 3) y robar 2 cartas.' },
    { spec: 'ismcts', label: 'ISMCTS (S&B 8.11)', family: 'búsqueda', desc: 'MCTS para información imperfecta: árbol de jugadas propias con UCB sobre manos posibles de los rivales (que juegan como Afinado), 2 jugadas de profundidad, 120 iteraciones.' },
    { spec: 'experto', label: 'Experto', family: 'búsqueda', desc: 'Monte Carlo: para las 6 mejores jugadas según Afinado, reparte 8 veces manos posibles a los rivales, simula una vuelta de mesa y evalúa cómo quedó contra el mejor rival.' },
  ];
  function describe(spec, models) {
    const hit = CATALOG.find(a => a.spec === spec);
    if (hit) return hit;
    const [kind, arg] = spec.split(/:(.*)/s);
    if (MODEL_KINDS.includes(kind)) {
      const m = models && models[arg];
      return { spec, label: (m && m.label) || arg, family: (m && m.family) || 'rl', desc: (m && m.desc) || '' };
    }
    if (kind === 'ismcts' && arg) { const [it, k = 6, d = 2] = arg.split(':').map(Number); return { spec, label: `ISMCTS ${it} (S&B 8.11)`, family: 'búsqueda', desc: `MCTS para información imperfecta: ${it} iteraciones, ${k} candidatas, ${d} jugadas propias de profundidad.` }; }
    if (kind === 'experto' && arg) { const [r, k = 6, h = 1] = arg.split(':').map(Number); return { spec, label: `Experto (${r}×${k}${h > 1 ? `, ${h} vueltas` : ''})`, family: 'búsqueda', desc: `Monte Carlo: ${r} simulaciones de ${h} vuelta${h > 1 ? 's' : ''} de mesa para cada una de las ${k} mejores jugadas según Afinado.` }; }
    throw new Error(`Agente desconocido: ${spec}`);
  }
  function makeAgent(spec, models) {
    const [kind, arg] = spec.split(/:(.*)/s);
    if (kind === 'experto') return arg ? experto(...arg.split(':').map(Number)) : experto();
    if (kind === 'ismcts') return arg ? ismcts(...arg.split(':').map(Number)) : ismcts();
    if (MODEL_KINDS.includes(kind)) {
      const m = models && models[arg];
      if (!m) throw new Error(`Falta el modelo ${arg}`);
      return kind === 'tabla' ? tablaAgent(m) : kind === 'lineal' ? linealAgent(m) : kind === 'valor' ? valorAgent(m) : netAgent(m);
    }
    if (H[kind]) return H[kind];
    throw new Error(`Agente desconocido: ${spec}`);
  }

  // ---------- partida ----------
  // agents[i].act(game, rng) -> acción
  function playGame(agents, rng) {
    const g = new Game(agents.length, 0, rng);
    while (!g.done) g.step(agents[g.cur].act(g, g.rng));
    return { score: g.ranks(), winners: g.winners, turns: g.turn, rounds: g.round, birds: g.cols.map(c => c.reduce((a, b) => a + b, 0)), why: g.endReason };
  }

  // ---------- Elo (cubirds/arena/elo.py) ----------
  // Cada partida se parte en duelos de a pares; menor puesto gana. Los duelos
  // entre dos entradas del mismo agente (eliminatoria) no cuentan.
  const ELO_INITIAL = 1500, ELO_K = 32;
  const expected = (ra, rb) => 1 / (1 + Math.pow(10, (rb - ra) / 400));
  function eloUpdate(ratings, seats, scores, k = ELO_K) {
    const n = seats.length, d = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      if (i === j || seats[i] === seats[j]) continue;
      const s = scores[i] < scores[j] ? 1 : scores[i] === scores[j] ? 0.5 : 0;
      d[i] += s - expected(ratings[seats[i]], ratings[seats[j]]);
    }
    for (let i = 0; i < n; i++) ratings[seats[i]] += d[i] * k / (n - 1);
  }

  // ---------- torneos (cubirds/arena/tournament.py) ----------
  function splitEven(n, tables) {
    const base = Math.floor(n / tables), extra = n % tables;
    return Array.from({ length: tables }, (_, i) => base + (i < extra ? 1 : 0));
  }
  // Cuadro de eliminatoria: en cada ronda se sortea el tamaño de mesa (2 a 5) y pasa el ganador.
  function randomPlan(nPlayers, rng, lo = 2, hi = MAX_TABLE) {
    const plan = [];
    let n = nPlayers;
    while (n > 1) {
      if (n <= hi) { plan.push({ players: n, tables: [n] }); break; }
      const size = lo + rng.int(hi - lo + 1);
      const sizes = splitEven(n, Math.ceil(n / size)).filter(x => x > 0);
      plan.push({ players: n, tables: sizes });
      n = sizes.length;
    }
    return plan;
  }

  function Recorder(n, totalGames) {
    const ratings = new Array(n).fill(ELO_INITIAL);
    const st = Array.from({ length: n }, () => ({ games: 0, wins: 0, turns: 0, birds: 0, fair: 0 }));
    const ahead = Array.from({ length: n }, () => new Array(n).fill(0));
    const together = Array.from({ length: n }, () => new Array(n).fill(0));
    const every = Math.max(1, Math.floor(totalGames / 120));
    const history = [[0, ratings.map(r => +r.toFixed(1))]];
    let games = 0;
    return {
      st,
      get games() { return games; },
      record(seats, res) {
        const score = res.score;
        eloUpdate(ratings, seats, score);
        const nw = score.filter(x => x === 0).length;
        seats.forEach((a, ia) => {
          const s = st[a];
          s.games++; s.turns += res.turns; s.birds += res.birds[ia]; s.fair += 1 / seats.length;
          if (score[ia] === 0) s.wins += 1 / nw;
          seats.forEach((b, ib) => {
            if (a === b) return;
            together[a][b]++;
            ahead[a][b] += score[ia] < score[ib] ? 1 : score[ia] === score[ib] ? 0.5 : 0;
          });
        });
        games++;
        if (games % every === 0) history.push([games, ratings.map(r => +r.toFixed(1))]);
      },
      snapshot(info, extra) {
        const hist = history.slice();
        if (hist[hist.length - 1][0] !== games) hist.push([games, ratings.map(r => +r.toFixed(1))]);
        return {
          agents: info.map((a, i) => {
            const s = st[i], gg = Math.max(s.games, 1);
            return Object.assign({
              id: i, spec: a.spec, label: a.label, family: a.family, desc: a.desc,
              elo: +ratings[i].toFixed(1), games: s.games,
              win_rate: +(s.wins / gg).toFixed(4),
              // ganar 1 de cada n en mesas de n es lo esperable: >1 es mejor que el promedio
              win_index: +(s.wins / Math.max(s.fair, 1e-9)).toFixed(3),
              avg_turns: +(s.turns / gg).toFixed(1), avg_birds: +(s.birds / gg).toFixed(2),
            }, extra ? extra(i) : {});
          }),
          history: hist,
          h2h: { ahead: ahead.map(r => r.map(x => +x.toFixed(1))), together: together.map(r => r.slice()) },
        };
      },
    };
  }

  function envelope(t0, name, format, config, played, body) {
    return Object.assign({
      id: new Date(t0).toISOString().replace(/[-:]/g, '').slice(0, 15).replace('T', '-'),
      name, format, created: new Date(t0).toISOString(), engine: 'browser',
      config: Object.assign(config, { k: ELO_K }),
      played, seconds: +((Date.now() - t0) / 1000).toFixed(1), notes: [],
    }, body);
  }

  // Liga: cada partida sienta a los que menos jugaron; tamaño de mesa fijo o 'variada' (2 a 5).
  function* tournament({ specs, games = 200, tableSize = 4, seed, models, name }) {
    seed = seed ?? Math.floor(Math.random() * 2 ** 31);
    const rng = makeRng(seed);
    const n = specs.length;
    if (n < 2) throw new Error('Elegí al menos 2 agentes');
    const varied = tableSize === 'variada';
    const info = specs.map(s => describe(s, models));
    const agents = specs.map(s => makeAgent(s, models));
    const rec = Recorder(n, games);
    const t0 = Date.now(), log = [];
    const snapshot = g => envelope(t0, name || `Liga de ${n} agentes`, 'liga',
      { games, table_size: tableSize, seed }, g, Object.assign(rec.snapshot(info), { games_log: log.slice() }));
    for (let g = 1; g <= games; g++) {
      const size = Math.min(n, varied ? 2 + rng.int(MAX_TABLE - 1) : Math.max(2, Math.min(tableSize, MAX_TABLE)));
      const order = [...Array(n).keys()].map(i => [rec.st[i].games, rng(), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(x => x[2]);
      const seats = rng.shuffle(order.slice(0, size));
      const res = playGame(seats.map(i => agents[i]), rng);
      rec.record(seats, res);
      log.push({ s: seats, w: res.winners, t: res.turns, b: res.birds });
      yield { g, total: games, snapshot };
    }
  }

  // Eliminatoria repetida: mesas de 2 a 5 sorteadas en cada ronda, pasa el ganador
  // (si empatan, se sortea entre los empatados).
  function* knockout({ specs, entries = 1, repetitions = 20, seed, models, name, tableRange = [2, 5] }) {
    seed = seed ?? Math.floor(Math.random() * 2 ** 31);
    const rng = makeRng(seed);
    const n = specs.length;
    if (n < 2) throw new Error('Elegí al menos 2 agentes');
    const field = [];
    for (let a = 0; a < n; a++) for (let e = 0; e < entries; e++) field.push(a);
    const plans = Array.from({ length: repetitions }, () => randomPlan(field.length, rng, tableRange[0], tableRange[1]));
    const total = plans.reduce((acc, plan) => acc + plan.reduce((a, r) => a + r.tables.filter(s => s > 1).length, 0), 0);
    const maxRounds = Math.max(...plans.map(p => p.length));
    const info = specs.map(s => describe(s, models));
    const agents = specs.map(s => makeAgent(s, models));
    const rec = Recorder(n, total);
    const reach = Array.from({ length: n }, () => new Array(maxRounds).fill(0));
    const finals = new Array(n).fill(0), titles = new Array(n).fill(0);
    let reps = 0;
    const t0 = Date.now(), brackets = [];
    const snapshot = g => {
      const app = Math.max(1, entries * reps);
      return envelope(t0, name || `Eliminatoria de ${field.length} jugadores`, 'eliminatoria',
        { games: total, seed, table_range: tableRange, entries, repetitions, repetitions_done: reps },
        g, Object.assign(rec.snapshot(info, a => ({
          title_rate: +(titles[a] / Math.max(reps, 1)).toFixed(4),
          final_rate: +(finals[a] / app).toFixed(4),
          reach: reach[a].map(x => +(x / app).toFixed(4)),
          avg_round: +(reach[a].reduce((s, x) => s + x, 0) / app).toFixed(3),
        })), { brackets: brackets.slice() }));
    };
    let g = 0;
    for (let rep = 0; rep < repetitions; rep++) {
      const plan = plans[rep];
      const repReach = Array.from({ length: n }, () => new Array(maxRounds).fill(0));
      let alive = field.map((a, i) => ({ a, i }));
      let champion = -1;
      const repLog = [];
      for (let r = 0; r < plan.length; r++) {
        const { tables } = plan[r];
        const tablesLog = [];
        rng.shuffle(alive);
        for (const x of alive) repReach[x.a][r]++;
        const next = [];
        let pos = 0;
        for (const size of tables) {
          const table = alive.slice(pos, pos + size);
          pos += size;
          const ent = table.map(x => [x.a, x.i % entries]);
          if (size === 1) { next.push(...table); tablesLog.push({ e: ent, bye: true }); continue; }
          const res = playGame(table.map(x => agents[x.a]), rng);
          rec.record(table.map(x => x.a), res);
          const win = rng.pick(res.winners);
          next.push(table[win]);
          tablesLog.push({ e: ent, w: res.winners, adv: [win], t: res.turns, b: res.birds });
          if (tables.length === 1) champion = table[win].a;
          g++;
          if (g < total) yield { g, total, snapshot };
        }
        alive = next;
        repLog.push(tablesLog);
      }
      brackets.push(repLog);
      for (let a = 0; a < n; a++) {
        for (let r = 0; r < plan.length; r++) reach[a][r] += repReach[a][r];
        finals[a] += repReach[a][plan.length - 1];
      }
      titles[champion]++;
      reps++;
      if (g === total) yield { g, total, snapshot };
    }
  }

  const api = {
    SPECIES, COUNTS, SMALL, BIG, NS, NROWS, HAND, A_NODRAW, A_DRAW, A_FLOCK0, A_NOFLOCK, NACTIONS, MAX_TABLE,
    playAction, decodePlay, makeRng, hasWon, enclosed, takenBy, Game, progress, handValue, flockChoice, H, DEFAULT, AFINADO, coleccionistaScores, ranked, determinize, experto, ismcts, evaluate, EXPERTO_W,
    G_DIM, A_DIM, DIM, globalFeatures, actionFeatures, decision, netLogit, netValue, netAgent, sbKey, tablaAgent, linealAgent, valorAgent, modelKinds: MODEL_KINDS, loadModels,
    CATALOG, describe, makeAgent, playGame, eloUpdate, ELO_INITIAL, ELO_K, tournament, knockout, randomPlan,
  };
  root.CuBirdsEngine = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;

  // ---------- modo worker ----------
  const isWorker = typeof WorkerGlobalScope !== 'undefined' && root instanceof WorkerGlobalScope;
  if (isWorker) {
    let stop = false;
    root.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'stop') { stop = true; return; }
      if (m.type !== 'run') return;
      stop = false;
      try {
        const it = m.format === 'eliminatoria' ? knockout(m) : tournament(m);
        let last = 0, step;
        const tick = () => {
          const t = Date.now();
          while (!(step = it.next()).done) {
            const { g, total, snapshot } = step.value;
            if (stop || g === total) { root.postMessage({ type: 'done', result: snapshot(g), stopped: stop && g !== total }); return; }
            if (Date.now() - last > 250) { last = Date.now(); root.postMessage({ type: 'progress', result: snapshot(g) }); }
            if (Date.now() - t > 40) { setTimeout(tick, 0); return; } // deja pasar el mensaje de 'stop'
          }
        };
        tick();
      } catch (err) {
        root.postMessage({ type: 'error', message: String(err && err.message || err) });
      }
    };
  }
})(typeof self !== 'undefined' ? self : this);
