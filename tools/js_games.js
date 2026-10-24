// Juega partidas con el motor JS y escribe el log de cada una (para tests/test_js_parity.py).
// Uso: node tools/js_games.js '[[seed, ["goloso","azar"]], ...]'
const E = require('../web/engine.js');
const jobs = JSON.parse(process.argv[2]);
const out = jobs.map(([seed, specs]) => {
  const g = new E.Game(specs.length, seed);
  const agents = specs.map(s => E.makeAgent(s));
  while (!g.done) g.step(agents[g.cur].act(g, g.rng));
  return { log: g.log, winners: g.winners, turns: g.turn };
});
process.stdout.write(JSON.stringify(out));
