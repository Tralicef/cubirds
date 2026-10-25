"""
Publica un modelo entrenado en web/arena/models para que juegue en el sitio.

    python tools/export_model.py runs/ppo1/best.json ppo "PPO" "PPO por jugada: ..." [--family rl]
"""

import argparse
import json
import shutil
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "web" / "arena" / "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("id")
    ap.add_argument("label")
    ap.add_argument("desc")
    ap.add_argument("--family", default="rl")
    a = ap.parse_args()
    MODELS.mkdir(parents=True, exist_ok=True)
    kind = json.loads(Path(a.src).read_text())["kind"]
    file = f"{a.id}.json"
    shutil.copy(a.src, MODELS / file)
    idx_path = MODELS / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else []
    prev = next((m for m in idx if m["id"] == a.id), {})
    idx = [m for m in idx if m["id"] != a.id]
    entry = {"id": a.id, "file": file, "kind": kind, "label": a.label, "family": a.family, "desc": a.desc}
    if prev.get("variants"):
        entry["variants"] = prev["variants"]  # p. ej. el mismo modelo con búsqueda (az:<id>:48)
    idx.append(entry)
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=1))
    print(f"{kind}:{a.id} → {MODELS / file}")


if __name__ == "__main__":
    main()
