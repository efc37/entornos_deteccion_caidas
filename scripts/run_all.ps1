# Ejecuta el pipeline completo en Windows (PowerShell), desde la raíz del repositorio:
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
# Requiere haber instalado las dependencias (ver README) y, para Le2i, haber
# descomprimido el dataset en data\raw\le2i (descarga manual).
$ErrorActionPreference = "Stop"

Write-Host "1) Descarga de URFD (se omite lo ya descargado)"
python scripts/download_urfd.py

Write-Host "2) Comprobación de datos"
python -m src.datasets --check

Write-Host "3) Extracción de esqueletos con YOLOv8-Pose (GPU 0)"
python -m src.extract_skeletons --dataset urfd le2i --device 0

Write-Host "4) Experimentos E1-E6 (tablas y figuras en results\)"
python -m src.experiments

Write-Host "5) Modelos finales para el servicio (entrenados con URFD + Le2i)"
python -m src.train_baseline --train urfd le2i --model rf
python -m src.train_deep_model --train urfd le2i --arch gru

Write-Host "6) Demostración del servicio sobre una caída y una ADL de URFD"
python -m src.service.demo_service --model results/models/gru_urfd+le2i.pt `
    --seq data/skeletons/urfd/fall-01.npz data/skeletons/urfd/adl-01.npz --respuesta ninguna

Write-Host "Hecho. Resumen en results\summary.md"
