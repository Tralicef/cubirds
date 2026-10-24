#!/bin/sh
# Ranking oficial con todos los agentes: bots, búsqueda y todos los modelos de web/arena/models.
#   sh tools/ranking_oficial.sh [partidas]
set -e
cd "$(dirname "$0")/.."
N=${1:-4000}
SPECS="azar goloso coleccionista afinado experto ismcts"
SPECS="$SPECS $(node -e "const m=require('./web/arena/models/index.json');console.log(m.map(x=>x.kind+':'+x.id).join(' '))")"
node web/arena/rank.js "$N" $SPECS
