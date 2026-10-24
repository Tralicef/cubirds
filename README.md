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
- `web/`: el sitio, publicado en https://cubirds.vercel.app.

## Probarlo

```bash
python3 -m pytest -q
sh web/build.sh && python3 -m http.server 8766 --directory web/dist
```

Vercel despliega solo cada push a `main` (carpeta raíz del proyecto: `web/`; `web/build.sh` arma `web/dist`).
