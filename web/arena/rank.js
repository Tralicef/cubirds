// Ranking oficial: juega una liga con mesas de 2 a 5 jugadores sorteadas en cada
// partida (sentando primero a los que jugaron menos) y calcula el Elo de cada agente
// como promedio del Elo secuencial sobre 50 órdenes al azar de las partidas (no
// depende del orden), con un margen 95% por bootstrap de partidas.
// Escribe web/arena/ratings.json (lo usa el juego para elegir rival) y guarda la
// liga en web/arena/data para la pestaña Torneos.
//   node web/arena/rank.js <partidas> spec1 spec2 ...
const fs = require('fs'), path = require('path');
const E = require(path.join(__dirname, '..', 'engine.js'));
const [gamesArg, ...specs] = process.argv.slice(2);
const N = +gamesArg, n = specs.length;
const models = E.loadModels ? E.loadModels(path.join(__dirname, 'models')) : {};
const info = specs.map(s => E.describe(s, models));
const agents = specs.map(s => E.makeAgent(s, models));

const rng = E.makeRng(2026), t0 = Date.now();
const played = new Array(n).fill(0), games = [];
for (let g = 0; g < N; g++) {
  const size = Math.min(n, 2 + rng.int(E.MAX_TABLE - 1));
  const seats = rng.shuffle([...specs.keys()].map(i => [played[i], rng(), i]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).slice(0, size).map(x => x[2]));
  const res = E.playGame(seats.map(i => agents[i]), rng);
  seats.forEach(i => played[i]++);
  games.push({ s: seats, r: res.score, w: res.winners, t: res.turns, b: res.birds });
  if ((g + 1) % 50 === 0) process.stdout.write(`\r${g + 1}/${N} partidas · ${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
console.log();

const erng = E.makeRng(7);
function eloOf(list) {
  const r = new Array(n).fill(E.ELO_INITIAL);
  for (const g of list) E.eloUpdate(r, g.s, g.r);
  return r;
}
function stableElo(list, perms) {
  const acc = new Array(n).fill(0);
  for (let p = 0; p < perms; p++) eloOf(erng.shuffle(list.slice())).forEach((v, i) => acc[i] += v / perms);
  return acc;
}
const elo = stableElo(games, 50);
const boot = Array.from({ length: 30 }, () => stableElo(Array.from({ length: games.length }, () => games[erng.int(games.length)]), 3));
const ci = elo.map((_, i) => { const v = boot.map(b => b[i]), m = v.reduce((a, b) => a + b, 0) / v.length; return 1.96 * Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length); });

const st = specs.map(() => ({ games: 0, wins: 0, fair: 0, birds: 0, turns: 0 }));
const ahead = specs.map(() => new Array(n).fill(0)), together = specs.map(() => new Array(n).fill(0));
for (const g of games) g.s.forEach((a, k) => {
  const s = st[a];
  s.games++; s.fair += 1 / g.s.length; s.birds += g.b[k]; s.turns += g.t;
  if (g.r[k] === 0) s.wins += 1 / g.w.length;
  g.s.forEach((b, j) => { if (a === b) return; together[a][b]++; ahead[a][b] += g.r[k] < g.r[j] ? 1 : g.r[k] === g.r[j] ? 0.5 : 0; });
});
const stats = i => {
  const s = st[i], gg = Math.max(1, s.games);
  return { games: s.games, win_rate: +(s.wins / gg).toFixed(3), win_index: +(s.wins / Math.max(s.fair, 1e-9)).toFixed(3), avg_birds: +(s.birds / gg).toFixed(2), avg_turns: +(s.turns / gg).toFixed(1) };
};
const ranked = specs.map((spec, i) => ({ spec, label: info[i].label, family: info[i].family, desc: info[i].desc, elo: Math.round(elo[i]), ci: Math.round(ci[i]), ...stats(i) }))
  .sort((a, b) => b.elo - a.elo);
const method = 'Elo multijugador (K=32, duelos por pares: el que ganó la partida le gana a cada uno de los demás; los que no ganaron empatan entre sí) promediado sobre 50 órdenes de partidas; margen 95% por bootstrap. Mesas de 2 a 5 jugadores sorteadas en cada partida.';
fs.writeFileSync(path.join(__dirname, 'ratings.json'), JSON.stringify({ updated: new Date().toISOString(), games: N, method, agents: ranked }, null, 1));
ranked.forEach((a, k) => console.log(String(k + 1).padStart(2), a.label.padEnd(36), a.elo, '±' + a.ci, `${a.games} pj`, `gana ${(a.win_rate * 100).toFixed(1)}%`, `índice ${a.win_index}`));

// También como torneo para la pestaña Torneos (ranking, evolución, cara a cara, partidas).
const r = new Array(n).fill(E.ELO_INITIAL), every = Math.max(1, Math.floor(N / 120)), history = [[0, r.slice()]];
games.forEach((g, k) => { E.eloUpdate(r, g.s, g.r); if ((k + 1) % every === 0 || k + 1 === N) history.push([k + 1, r.map(x => +x.toFixed(1))]); });
const now = new Date(), id = now.toISOString().replace(/[-:]/g, '').slice(0, 15).replace('T', '-');
const name = process.env.TORNEO || 'Ranking oficial · mesas de 2 a 5';
const t = {
  id, name, format: 'liga', engine: 'browser', created: now.toISOString(), seconds: +((Date.now() - t0) / 1000).toFixed(1), played: N,
  config: { games: N, table_size: '2 a 5', k: E.ELO_K },
  agents: specs.map((spec, i) => ({ id: i, spec, label: info[i].label, family: info[i].family, desc: info[i].desc, elo: Math.round(elo[i]), ci: Math.round(ci[i]), ...stats(i) })),
  history, h2h: { ahead, together }, games_log: games.map(g => ({ s: g.s, w: g.w, t: g.t, b: g.b })),
  notes: ['Elo = promedio del Elo secuencial sobre 50 órdenes al azar de las partidas (no depende del orden); la línea de evolución es un solo orden, el real.', 'Índice de victorias: victorias sobre las esperables por azar (1 de cada n en mesas de n). Más de 1 es mejor que el promedio.'],
};
const out = path.join(__dirname, 'data');
fs.mkdirSync(out, { recursive: true });
const file = `${id}.json`;
fs.writeFileSync(path.join(out, file), JSON.stringify(t));
const idxFile = path.join(out, 'index.json');
const prev = fs.existsSync(idxFile) ? JSON.parse(fs.readFileSync(idxFile)) : [];
// el ranking anterior con el mismo nombre se reemplaza
for (const e of prev) if (e.name === name && fs.existsSync(path.join(out, e.file))) fs.unlinkSync(path.join(out, e.file));
const index = prev.filter(e => e.name !== name);
index.unshift({ id, file, name, created: t.created, engine: 'browser', format: 'liga', games: N, table_size: '2 a 5', agents: n });
fs.writeFileSync(idxFile, JSON.stringify(index, null, 1));
console.log(`Torneo guardado: web/arena/data/${file}`);
