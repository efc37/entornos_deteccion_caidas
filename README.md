# Detección de caídas mediante visión por computador (Trabajo 17)

Entornos Inteligentes (AAL) · Universidad de Alicante · Práctica 3, grupo 3
Joël Expósito · Elena Fernández · Alejandro Operé

Servicio de una **casa que te cuida** que detecta posibles caídas en una habitación a
partir de la **pose (esqueleto)** extraída localmente con YOLOv8-Pose. Las distingue de
actividades cotidianas con un **modelo temporal** (GRU / ST-GCN), reduce las falsas
alarmas con **persistencia, comprobación de inactividad y confirmación verbal**, y
**no guarda ni transmite imágenes**.

```
cámara ─► YOLOv8n-Pose ─► esqueleto (17 pts) ─► ventana 1,3 s ─► GRU / ST-GCN ─► p(caída)
              └─ la imagen se descarta                                              │
     persistencia ─► ¿sigue en el suelo 2 s? ─► "¿Se encuentra bien?" ─► aviso a contacto
```

La justificación completa (estado del arte, decisiones, datos, protocolo, integración y
referencias) está en [`docs/metodologia.md`](docs/metodologia.md).

## Estructura

```
├── scripts/
│   ├── download_urfd.py        descarga URFD desde la web oficial
│   └── run_all.ps1 / .sh       pipeline completo de principio a fin
├── src/
│   ├── config.py               parámetros (rutas, ventanas, umbrales, modelos)
│   ├── datasets.py             indexado de URFD/Le2i y etiquetas por frame
│   ├── extract_skeletons.py    YOLOv8-Pose -> .npz (solo coordenadas)
│   ├── preprocessing.py        limpieza, remuestreo a 25 fps, ventanas, normalización
│   ├── baseline_features.py    25 features geométricas/cinemáticas + reglas
│   ├── train_baseline.py       método base: reglas, Random Forest, SVM
│   ├── models_deep.py          GRU, ST-GCN y aumento de datos
│   ├── train_deep_model.py     entrenamiento de los modelos profundos
│   ├── postprocess.py          persistencia + confirmación de inactividad
│   ├── evaluation.py           métricas por ventana y por evento
│   ├── experiments.py          experimentos E1-E6, tablas y figuras
│   ├── synthetic.py            datos sintéticos (solo para probar el código)
│   └── service/
│       ├── fall_service.py     máquina de estados del servicio
│       └── demo_service.py     demo sobre secuencias o con webcam
├── tests/test_pipeline.py      pruebas automáticas
├── docs/metodologia.md         justificación y base del póster
├── data/README.md              origen de los datos y licencias
└── results/                    tablas (csv, summary.md) y figuras generadas
```

## Instalación (Windows, GPU NVIDIA)

```powershell
cd entornos_deteccion_caidas
python -m venv .venv
.venv\Scripts\activate
# 1) PyTorch con CUDA PRIMERO (versión de CUDA según https://pytorch.org/get-started/locally/)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# 2) Resto de dependencias
pip install -r requirements.txt
# 3) Comprobación
python -c "import torch; print(torch.cuda.is_available())"   # debe imprimir True
python -m pytest -q                                          # pruebas con datos sintéticos
```

## Datos

Los datasets **no se incluyen** en el repositorio (licencias). Detalles en
[`data/README.md`](data/README.md).

1. **URFD** (CC BY-NC-SA 4.0), descarga automática (~2-3 GB):
   ```powershell
   python scripts/download_urfd.py            # o:  --format mp4   (más ligero)
   ```
2. **Le2i**: la web oficial (https://imvia.ube.fr/?p=346) no tiene ahora enlace activo.
   Descargad una copia pública (p. ej. Kaggle, "Le2i fall detection") y descomprimidla
   en `data/raw/le2i/`. El código localiza solo los vídeos y sus `Annotation_files`.
3. Comprobación: `python -m src.datasets --check`

## Ejecución

Todo de una vez: `powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1`

O paso a paso:

```powershell
# 1. Esqueletos (GPU 0). Solo se guardan coordenadas en data/skeletons/
python -m src.extract_skeletons --dataset urfd le2i --device 0

# 2. Experimentos E1-E6 -> results/summary.md, results/*.csv, results/figures/*.png
python -m src.experiments
#    variantes: --models rules rf gru   |   --quick (pocas épocas)   |   --no-occlusion

# 3. Modelos finales para el servicio
python -m src.train_baseline   --train urfd le2i --model rf
python -m src.train_deep_model --train urfd le2i --arch gru

# 4. Demostración del servicio (respuesta simulada: ok | ayuda | ninguna)
python -m src.service.demo_service --model results/models/gru_urfd+le2i.pt `
       --seq data/skeletons/urfd/fall-01.npz data/skeletons/urfd/adl-01.npz --respuesta ninguna
#    en directo con la webcam (los fotogramas se descartan):
python -m src.service.demo_service --model results/models/gru_urfd+le2i.pt --webcam 0
```

Sin datos reales se puede probar todo el flujo con esqueletos sintéticos (los
resultados **no tienen valor científico** y no deben usarse en el póster):

```powershell
python -m src.experiments --synthetic        # -> results_synthetic/
python -m src.service.demo_service --synthetic
```

Tiempo orientativo con GPU: extracción ~10-20 min, experimentos ~15-30 min (la SVM en
Le2i es la parte más lenta en CPU).

## Experimentos

| Id | Pregunta | Protocolo |
|---|---|---|
| E1 | ¿Distingue caídas de ADL? | URFD, 5 particiones agrupadas por secuencia |
| E2 | ¿Funciona en una habitación nueva? | Le2i, una escena fuera cada vez |
| E3 | ¿Generaliza a otra cámara y dataset? | URFD → Le2i y Le2i → URFD, sin reajuste |
| E4 | ¿Cuántas falsas alarmas genera? | Streaming: umbral / + persistencia / + inactividad |
| E5 | ¿Cómo afectan el campo de visión y las oclusiones? | F1 por escena y con articulaciones ocultas |
| E6 | ¿Qué conserva y cuánto cuesta cada representación? | bytes/s, ms/frame, información personal |

Comparación: **método base** (reglas, Random Forest, SVM sobre features) frente a la
**propuesta** (GRU y ST-GCN sobre la secuencia; GRU con y sin aumento de datos).

## Resultados

*Pendiente de ejecutar con los datos reales.* Tras `python -m src.experiments`, las
tablas quedan en [`results/summary.md`](results/summary.md) y las figuras para el póster
en `results/figures/`.

## Privacidad

* La imagen solo existe en memoria durante la inferencia de pose; en disco hay
  coordenadas (experimentos) o eventos (servicio).
* Ni los datos originales ni los esqueletos derivados se versionan.
* Quitar la imagen **no equivale a anonimizar**: el esqueleto conserva postura,
  proporciones y forma de andar. Ver la discusión en `docs/metodologia.md`.

## Licencia y citas

Código bajo la licencia del repositorio (`LICENSE`). Los datos pertenecen a sus
autores: URFD (Kwolek y Kepski, 2014; CC BY-NC-SA 4.0) y Le2i (Charfi et al., 2013).
YOLOv8-Pose es de Ultralytics (AGPL-3.0).
