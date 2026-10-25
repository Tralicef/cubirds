# CuBirds

El juego de cartas CuBirds (Stefan Alexander, 2018) en Python y en el navegador,
con bots, agentes de aprendizaje por refuerzo y una arena con Elo. Es el mismo
ejercicio que [Toma 6](https://toma6.vercel.app), con otro juego.

## Reglas en una línea

Jugás todas las cartas de una especie en una punta de una fila; si encerrás pájaros
entre los tuyos y otro igual, te los llevás. Con suficientes cartas de una especie
formás una bandada y guardás 1 o 2 en tu colección. Gana el primero con 7 especies
distintas o 2 especies con 3 pájaros.

| Especie   | Cartas | Bandada chica | Bandada grande |
|-----------|-------:|--------------:|---------------:|
| Flamenco  |      7 |             2 |              3 |
| Búho      |     10 |             3 |              4 |
| Tucán     |     10 |             3 |              4 |
| Pato      |     13 |             4 |              6 |
| Loro      |     13 |             4 |              6 |
| Urraca    |     17 |             5 |              7 |
| Carricero |     20 |             6 |              9 |
| Petirrojo |     20 |             6 |              9 |

## Estructura

- `cubirds/game/engine.py`: reglas. Cada turno son hasta tres decisiones (jugar, robar, bandada) en un único espacio de 75 acciones.
- `cubirds/game/bots.py`: bots heurísticos.
- `web/engine.js`: el mismo motor en JavaScript, con el mismo generador aleatorio. `tests/test_js_parity.py` comprueba que las partidas sean idénticas en los dos lenguajes.
- `web/`: el sitio, publicado en https://cu-birds.vercel.app:
  - `index.html`: el juego, contra cualquier agente de la arena con su Elo.
  - `arena.html`: torneos (ligas y eliminatorias), ranking Elo y simulación en el navegador.
  - `lab.html` + `lab-worker.js`: 33 algoritmos aprendiendo en vivo, con curva, mapa de calor o pesos y una partida en cámara lenta.

## Bots

- **Azar**, **Goloso** (se lleva la mayor cantidad de pájaros) y **Coleccionista** (puntúa su mano según qué tan cerca está de formar bandadas).
- **Afinado**: Coleccionista con pesos encontrados por búsqueda aleatoria. Aprendió que conviene mucho más completar especies que ya tenés (camino a "2 especies con 3") que sumar especies nuevas.
- **ISMCTS** (sólo en JS): árbol de jugadas propias con UCB sobre manos posibles de los rivales, 2 jugadas de profundidad.
- **Experto**: Monte Carlo con determinización. Para las mejores jugadas según Afinado, reparte manos posibles a los rivales con las cartas ocultas, simula una vuelta de mesa y evalúa contra el mejor rival.

## Agentes que aprenden (`cubirds/rl`)

- `features.py`: cada decisión (jugar, robar, bandada) se describe con un vector global y uno por acción legal; la misma codificación está en `web/engine.js`.
- `ppo.py`: PPO por jugada, con shaping basado en potencial (Φ = 0,1·pájaros − 0,1·bandadas que faltan).
- `neuroevo.py`: la misma red entrenada con estrategias evolutivas.
- `alphazero.py`: AlphaZero adaptado a información imperfecta y varios jugadores. La búsqueda (PUCT, en `web/engine.js`) reparte manos posibles con las cartas ocultas en cada simulación, guarda el valor de cada jugador en cada nodo (max^n) y valúa las hojas con la red. El autojuego corre en Node (`tools/az_selfplay.js`) y el entrenamiento en torch. Arranca de los pesos de PPO.
- `sb/`: Sutton & Barto sobre afterstates. Tabulares (caps. 2, 5–8 y 12), aproximados (caps. 9–11 y 13) y TD-Gammon (16.1).

```bash
python -m cubirds.rl.ppo --iters 300 --out runs/ppo
python -m cubirds.rl.neuroevo --gens 150 --out runs/evo
python -m cubirds.rl.alphazero --iters 8 --games 600 --sims 48 --out runs/az
python -m cubirds.rl.sb entrenar --todos --episodios 10000 --exportar
python tools/export_model.py runs/ppo/best.json ppo "PPO" "PPO por jugada"
node web/arena/rank.js 4000 azar goloso coleccionista afinado experto red:ppo tabla:sb_sarsa ...
```

## Probarlo

```bash
python3 -m pytest -q
sh web/build.sh && python3 -m http.server 8766 --directory web/dist
```

Vercel despliega solo cada push a `main` (carpeta raíz del proyecto: `web/`; `web/build.sh` arma `web/dist`).
