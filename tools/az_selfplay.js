// Autojuego de AlphaZero: todos los asientos buscan con la misma red (web/engine.js: azSearch).
// Por cada decisión con más de una opción escribe una línea JSON con el estado compacto
// (para recalcular las características en Python), las visitas de cada acción legal
// y, al terminar la partida, la parte de la victoria de quien decidía.
// Con una liga (lista de agentes separados por comas) cada asiento menos uno es, con
// probabilidad 0,6, un agente de la liga en vez de la búsqueda; sólo se registran las
// decisiones de los asientos que buscan.
//   node tools/az_selfplay.js <modelo.json> <partidas> <semilla> <simulaciones> <salida.jsonl> [liga]
const fs = require('fs');
const E = require('../web/engine.js');
const [modelPath, gamesArg, seedArg, simsArg, out, leagueArg] = process.argv.slice(2);
const model = JSON.parse(fs.readFileSync(modelPath));
const league = leagueArg ? leagueArg.split(',').map(s => E.makeAgent(s, E.loadModels(require('path').join(__dirname, '..', 'web', 'arena', 'models')))) : [];
const rng = E.makeRng(+seedArg), sims = +simsArg, TEMP_MOVES = 8;
const fd = fs.openSync(out, 'w');
let decisions = 0;
for (let k = 0; k < +gamesArg; k++) {
  const n = 2 + rng.int(4);
  const g = new E.Game(n, rng.int(2 ** 30) + 1);
  const rows = [], moves = new Array(n).fill(0);
  const keep = rng.int(n), seat = Array.from({ length: n }, (_, q) => q !== keep && league.length && rng() < 0.6 ? league[rng.int(league.length)] : null);
  while (!g.done) {
    const legal = g.legalActions();
    if (legal.length === 1) { g.step(legal[0]); continue; }
    if (seat[g.cur]) { g.step(seat[g.cur].act(g, rng)); continue; }
    const s = E.azSearch(model, g, sims, rng, { noise: true });
    const tot = s.visits.reduce((a, b) => a + b, 0);
    rows.push({ s: JSON.parse(JSON.stringify({ n: g.n, rows: g.rows, hands: g.hands, cols: g.cols, cur: g.cur, phase: g.phase, deck: g.deck.length })), legal: s.legal, pi: s.visits.map(v => +(v / tot).toFixed(4)), p: g.cur });
    // temperatura 1 en las primeras jugadas de cada uno, después la más visitada
    let i;
    if (moves[g.cur] < TEMP_MOVES) { let u = rng() * tot; for (i = 0; i < s.visits.length - 1; i++) { u -= s.visits[i]; if (u < 0) break; } }
    else { i = 0; s.visits.forEach((v, j) => { if (v > s.visits[i]) i = j; }); }
    moves[g.cur]++;
    g.step(s.legal[i]);
  }
  for (const r of rows) {
    r.z = g.winners.includes(r.p) ? 1 / g.winners.length : 0;
    fs.writeSync(fd, JSON.stringify(r) + '\n');
  }
  decisions += rows.length;
}
fs.closeSync(fd);
console.log(JSON.stringify({ games: +gamesArg, decisions }));
