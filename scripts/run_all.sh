#!/usr/bin/env bash
# Pipeline completo (Linux/macOS). Desde la raíz del repositorio: bash scripts/run_all.sh
set -euo pipefail
python scripts/download_urfd.py
python -m src.datasets --check
python -m src.extract_skeletons --dataset urfd le2i --device 0
python -m src.experiments
python -m src.train_baseline --train urfd le2i --model rf
python -m src.train_deep_model --train urfd le2i --arch gru
python -m src.service.demo_service --model results/models/gru_urfd+le2i.pt \
    --seq data/skeletons/urfd/fall-01.npz data/skeletons/urfd/adl-01.npz --respuesta ninguna
echo "Hecho. Resumen en results/summary.md"
