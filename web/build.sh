#!/bin/sh
# Arma dist/ para servir en Vercel. Las páginas .html están en formato artifact
# de claude.ai (sin doctype ni <head>); acá se les agrega el esqueleto que falta.
set -e
cd "$(dirname "$0")"
rm -rf dist
mkdir -p dist
for page in *.html; do
  {
    cat <<'HEAD'
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#1F8A5B">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="CuBirds">
<style>
:root { color-scheme: light; padding-top: env(safe-area-inset-top, 0px); padding-bottom: env(safe-area-inset-bottom, 0px); }
body { margin: 0; font: 14px system-ui, -apple-system, sans-serif; }
img { max-width: 100%; }
[hidden] { display: none !important; }
</style>
HEAD
    cat "$page"
    echo '</html>'
  } > "dist/$page"
done
cp *.js dist/
if [ -d arena ]; then cp -r arena dist/; fi
