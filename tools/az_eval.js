// Evalúa agentes en mesas fijas (siempre las mismas partidas) contra Afinado, Coleccionista y Goloso.
// Índice de victorias: 1 = lo justo (ganar 1 de cada n en mesas de n).
//   node tools/az_eval.js <partidas> <modelo.json|-> spec1 spec2 ...   (el modelo se llama "cand")
const fs = require('fs'), path = require('path');
const E = require('../web/engine.js');
const [gamesArg, modelPath, ...specs] = process.argv.slice(2);
const models = E.loadModels(path.join(__dirname, '..', 'web', 'arena', 'models'));
if (modelPath !== '-') models.cand = JSON.parse(fs.readFileSync(modelPath));
const TABLES = [['afinado'], ['afinado', 'coleccionista'], ['afinado', 'goloso', 'coleccionista'], ['afinado', 'afinado', 'coleccionista', 'goloso']];
const out = {};
for (const spec of specs) {
  const X = E.makeAgent(spec, models);
  let idx = 0;
  for (let i = 0; i < +gamesArg; i++) {
    const rng = E.makeRng(90000 + i), tab = TABLES[i % TABLES.length].map(s => E.makeAgent(s)), n = tab.length + 1, me = Math.floor(i / TABLES.length) % n;
    const seats = tab.slice(); seats.splice(me, 0, X);
    const r = E.playGame(seats, rng);
    if (r.winners.includes(me)) idx += n / r.winners.length;
  }
  out[spec] = +(idx / +gamesArg).toFixed(3);
}
console.log(JSON.stringify(out));
