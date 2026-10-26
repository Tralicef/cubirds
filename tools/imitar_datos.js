// Datos para imitar a un bot: mesas de 2 a 5 con el bot en todos los asientos (cada asiento menos
// el primero es, con probabilidad 0,3, Afinado o PPO) y se registran sus decisiones en el formato
// de tools/az_selfplay.js, con la jugada elegida como única "visita".
//   node tools/imitar_datos.js <bot> <partidas> <semilla> <salida.jsonl>
const fs = require('fs'), path = require('path');
const E = require('../web/engine.js');
const [spec, gamesArg, seedArg, out] = process.argv.slice(2);
const models = E.loadModels(path.join(__dirname, '..', 'web', 'arena', 'models'));
const bot = E.makeAgent(spec, models), others = ['afinado', 'red:ppo'].map(s => E.makeAgent(s, models));
const rng = E.makeRng(+seedArg), fd = fs.openSync(out, 'w');
let decisions = 0;
for (let k = 0; k < +gamesArg; k++) {
  const n = 2 + rng.int(4), g = new E.Game(n, rng.int(2 ** 30) + 1);
  const seat = Array.from({ length: n }, (_, q) => q > 0 && rng() < 0.3 ? others[rng.int(2)] : bot);
  const rows = [];
  while (!g.done) {
    const legal = g.legalActions();
    if (legal.length === 1) { g.step(legal[0]); continue; }
    const a = seat[g.cur].act(g, g.rng);
    if (seat[g.cur] === bot) rows.push({ s: JSON.parse(JSON.stringify({ n: g.n, rows: g.rows, hands: g.hands, cols: g.cols, cur: g.cur, phase: g.phase, deck: g.deck.length })), legal, pi: legal.map(x => (x === a ? 1 : 0)), p: g.cur });
    g.step(a);
  }
  for (const r of rows) { r.z = g.winners.includes(r.p) ? 1 / g.winners.length : 0; fs.writeSync(fd, JSON.stringify(r) + '\n'); }
  decisions += rows.length;
}
fs.closeSync(fd);
console.log(JSON.stringify({ games: +gamesArg, decisions }));
