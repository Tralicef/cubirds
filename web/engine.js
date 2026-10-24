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
  const speciesWeight = (col, s) => col[s] === 0 ? 1.5 : col[s] < 3 ? 1.2 : 0.6;
  function handValue(hand, col) {
    let v = 0;
    for (let s = 0; s < NS; s++) {
      const k = hand[s];
      if (!k) continue;
      let base;
      if (k >= BIG[s]) base = 2.2;
      else if (k >= SMALL[s]) base = 1.0 + 0.6 * (k - SMALL[s]) / (BIG[s] - SMALL[s]);
      else base = 0.7 * k / SMALL[s];
      v += base * speciesWeight(col, s);
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
  const DRAW_BONUS = 0.35;
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
    coleccionista: bot((g, legal, rng) => {
      const h = g.hands[g.cur], col = g.cols[g.cur];
      const base = handValue(h, col);
      let best = [], bv = null;
      for (const a of legal) {
        const [s, r, side] = decodePlay(a);
        const taken = takenBy(g.rows[r], s, side);
        const nh = h.slice(); nh[s] = 0;
        for (const c of taken) nh[c]++;
        const v = handValue(nh, col) - base + (taken.length ? 0.05 * taken.length : DRAW_BONUS);
        if (bv === null || v > bv + 1e-9) { best = [a]; bv = v; }
        else if (v > bv - 1e-9) best.push(a);
      }
      return rng.pick(best);
    }),
  };

  // ---------- catálogo ----------
  const CATALOG = [
    { spec: 'azar', label: 'Azar', family: 'heurístico', desc: 'Elige cualquier jugada legal al azar.' },
    { spec: 'goloso', label: 'Goloso', family: 'heurístico', desc: 'Se lleva la mayor cantidad de pájaros posible; a igualdad, juega menos cartas. Siempre roba y forma la bandada más grande.' },
    { spec: 'coleccionista', label: 'Coleccionista', family: 'heurístico', desc: 'Puntúa su mano según qué tan cerca está de formar bandadas de especies que le faltan y elige la jugada que más la mejora.' },
  ];
  function describe(spec) {
    const hit = CATALOG.find(a => a.spec === spec);
    if (hit) return hit;
    throw new Error(`Agente desconocido: ${spec}`);
  }
  function makeAgent(spec, models) {
    const [kind] = spec.split(/:(.*)/s);
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
    const info = specs.map(describe);
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
    const info = specs.map(describe);
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
    playAction, decodePlay, makeRng, hasWon, enclosed, takenBy, Game, progress, handValue, flockChoice, H,
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
