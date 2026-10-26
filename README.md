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
  - `lab.html` + `lab-worker.js`: 34 algoritmos aprendiendo en vivo (de los bandidos a AlphaZero liviano), con curva, mapa de calor o pesos y una partida en cámara lenta.
  - `aprender.html`: curso de aprendizaje por refuerzo en 14 lecciones, de los bots escritos a mano (la búsqueda aleatoria de pesos de Afinado y el Estratega, que copia lo que enseñaron los mejores agentes) a AlphaZero: cada una muestra una jugada real paso a paso (cómo piensa y cómo aprende, con las cuentas), se puede entrenar en la página y trae su lugar en el ranking.

## Bots

- **Azar**, **Goloso** (se lleva la mayor cantidad de pájaros) y **Coleccionista** (puntúa su mano según qué tan cerca está de formar bandadas).
- **Afinado**: Coleccionista con pesos encontrados por búsqueda aleatoria. Aprendió que conviene mucho más completar especies que ya tenés (camino a "2 especies con 3") que sumar especies nuevas.
- **Estratega**: sigue lo que dejaron ver los mejores agentes (ver abajo). Con 2 cartas o menos no roba, así vacía la mano y cierra la ronda. También forma bandadas chicas de especies nuevas y prefiere llevarse especies raras. Sus pesos salieron de una búsqueda aleatoria de 160 combinaciones contra Afinado y LSTD. Con la regla de cerrar la ronda, la búsqueda dio vuelta la preferencia de Afinado: una especie nueva pasa a valer más que una empezada (2,32 contra 1,98). Le gana al AlphaZero original (1675 ± 55 contra 1622 ± 44 en el ranking oficial; Afinado tiene 1560) y empata con el que arrancó imitándolo (ver abajo).
- **ISMCTS** (sólo en JS): árbol de jugadas propias con UCB sobre manos posibles de los rivales, 2 jugadas de profundidad.
- **Experto**: Monte Carlo con determinización. Para las mejores jugadas según Afinado, reparte manos posibles a los rivales con las cartas ocultas, simula una vuelta de mesa y evalúa contra el mejor rival.

## Qué enseñan los mejores agentes

Salió de 3.000 partidas registradas jugada por jugada y de experimentos que cambian una sola conducta de un agente, en mesas de 3 contra dos Afinado (lo justo es ganar el 33%).

- **Cerrar la ronda.** Con 2 cartas o menos conviene no robar: vaciás la mano, los demás descartan las suyas y vos repartís y seguís. Con esa sola regla, Afinado pasa de 33% a 43,5%, PPO de 38% a 52% y AlphaZero con búsqueda de 46% a 51%. Ninguna red lo aprendió. La búsqueda de AlphaZero lo encontró a medias (con la mano vacía no roba el 34% de las veces), y obligarla a robar siempre la baja a 36%.
- **Bandadas apuradas y a lo ancho.** AlphaZero y PPO forman bandada apenas pueden, aunque sea chica: 6 por partida contra 3,3 de Afinado. Ganan por 7 especies distintas en el 61–67% de las partidas; Afinado y LSTD casi siempre ganan con 2 especies de 3 o más. Va junto con jugar con la mano chica (5,8 cartas contra 7,5) y llevarse cartas más seguido: a PPO, imponerle las bandadas de Afinado lo baja de 38% a 25%.
- **Especies raras.** Los modelos lineales (LSTD, SARSA, Q) premian llevarse Flamencos, Búhos y Tucanes y castigan llevarse Carriceros y Petirrojos.
- Ninguno rompe un grupo que ya alcanza para bandada, y casi siempre juegan la especie de la que tienen menos.

## AlphaZero desde el Estratega

El AlphaZero original perdía con el Estratega en duelo directo: en mesas de 2 a 5 con los dos (y Afinado, PPO o LSTD de relleno), ganaba el 29% de las partidas contra el 44% del Estratega. Le cerraban la ronda unas 1,6 veces por partida y perdía unas 10 cartas cada vez; él casi nunca dejaba de robar con la mano chica.

| Intento (mismas partidas) | Agente | Estratega |
|---|--:|--:|
| AlphaZero original, 48 simulaciones | 29% | 44% |
| … con la regla de cerrar la ronda impuesta | 32% | 41% |
| … 6 vueltas más en una liga con el Estratega | 32% | 43% |
| AlphaZero original, 192 simulaciones | 38% | 37% |
| Red que imita al Estratega, sin buscar | 37% | 39% |
| AlphaZero desde el Estratega, 48 simulaciones | 39% | 38% |
| AlphaZero desde el Estratega, 96 simulaciones | 40,5% | 35% |
| AlphaZero desde el Estratega, 192 simulaciones | 42% | 34% |

Entrenar más en liga mejoró mucho a AlphaZero contra Afinado (índice 1,54 → 1,89), pero casi nada contra el Estratega: partía de PPO, cuyos priors casi nunca dicen "no robar", y en su autojuego nadie cerraba la ronda. Lo que funcionó es lo que hizo AlphaGo con las partidas humanas: primero imitar. Una red aprendió de 300.000 decisiones del Estratega (coincide con él en el 81%) y después 8 vueltas de AlphaZero con 96 simulaciones, 3.200 partidas en una liga contra Estratega, Afinado y PPO, la llevaron a superarlo. PPO con el mismo ajuste en liga no mejoró.

En el ranking oficial (7.000 partidas) los cuatro primeros quedan a la par: AlphaZero desde el Estratega con 96 simulaciones 1678 ± 40, Estratega 1675 ± 55, con 48 simulaciones 1670 ± 35 y la red sola 1668 ± 44; el AlphaZero original sigue con 1622 ± 44 y PPO con 1617 ± 44.

## Agentes que aprenden (`cubirds/rl`)

- `features.py`: cada decisión (jugar, robar, bandada) se describe con un vector global y uno por acción legal; la misma codificación está en `web/engine.js`.
- `ppo.py`: PPO por jugada, con shaping basado en potencial (Φ = 0,1·pájaros − 0,1·bandadas que faltan).
- `neuroevo.py`: la misma red entrenada con estrategias evolutivas.
- `alphazero.py`: AlphaZero adaptado a información imperfecta y varios jugadores. La búsqueda (PUCT, en `web/engine.js`) reparte manos posibles con las cartas ocultas en cada simulación, guarda el valor de cada jugador en cada nodo (max^n) y valúa las hojas con la red. El autojuego corre en Node (`tools/az_selfplay.js`) y el entrenamiento en torch. Arranca de los pesos de PPO o de los que se le pasen con `--start`; con `--league` algunos asientos del autojuego son bots fijos.
- `imitacion.py`: una red que imita a un bot (los datos los juega `tools/imitar_datos.js`), como punto de partida para AlphaZero.
- `sb/`: Sutton & Barto sobre afterstates. Tabulares (caps. 2, 5–8 y 12), aproximados (caps. 9–11 y 13) y TD-Gammon (16.1).

```bash
python -m cubirds.rl.ppo --iters 300 --out runs/ppo
python -m cubirds.rl.neuroevo --gens 150 --out runs/evo
python -m cubirds.rl.alphazero --iters 8 --games 600 --sims 48 --out runs/az
python -m cubirds.rl.imitacion --bot estratega --games 4500 --out runs/clon_estratega.json
python -m cubirds.rl.alphazero --iters 8 --games 400 --sims 96 --start runs/clon_estratega.json --league estratega,estratega,afinado,red:ppo --out runs/az_clon
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
