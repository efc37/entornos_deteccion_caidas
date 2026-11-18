# Datos

Esta carpeta **no se versiona** (ver `.gitignore`): los datasets tienen licencias que
no permiten redistribuirlos desde este repositorio. Cada persona debe descargarlos
desde su fuente original.

```
data/
├── raw/
│   ├── urfd/            ← python scripts/download_urfd.py
│   │   ├── urfall-cam0-falls.csv      (etiquetas por frame de las caídas)
│   │   ├── fall-01-cam0-rgb/ ...      (PNG, cámara frontal)
│   │   └── adl-01-cam0-rgb/ ...
│   └── le2i/            ← descarga manual (ver abajo)
│       ├── Home_01/Videos/video (1).avi
│       ├── Home_01/Annotation_files/video (1).txt
│       └── Coffee_room_01/...
└── skeletons/           ← generado por python -m src.extract_skeletons (solo coordenadas)
```

## UR Fall Detection Dataset (URFD)
* Fuente: https://fenix.ur.edu.pl/mkepski/ds/uf.html (Universidad de Rzeszów).
* Licencia: **CC BY-NC-SA 4.0**, uso académico no comercial.
* Contenido usado: 30 caídas y 40 actividades cotidianas (ADL) de la cámara frontal
  (`cam0`, 30 fps, 640×480) y el fichero `urfall-cam0-falls.csv`, que etiqueta cada
  frame como *no tumbado* (−1), *cayendo* (0) o *tumbado en el suelo* (1).
* Descarga automática: `python scripts/download_urfd.py` (o `--format mp4`).
* Cita: B. Kwolek, M. Kepski. *Human fall detection on embedded platform using depth
  maps and wireless accelerometer.* Computer Methods and Programs in Biomedicine
  117(3):489–501, 2014.

## Le2i Fall Detection Dataset
* Origen: laboratorio Le2i / ImViA, Université de Bourgogne
  (https://imvia.ube.fr/?p=346). 191 vídeos, 25 fps, 320×240, una cámara, cuatro
  escenarios (Home, Coffee room, Office, Lecture room), con oclusiones por mobiliario
  y cambios de iluminación.
* La web oficial indica actualmente que el enlace de descarga está pendiente de
  renovarse, por lo que hay que usar una copia pública (p. ej. en Kaggle, buscando
  "Le2i fall detection"). Revise las condiciones de la copia que use; la web oficial
  no especifica licencia: úsese solo con fines académicos y citando el trabajo original.
* Descomprima en `data/raw/le2i/`. El código busca recursivamente los vídeos y su
  anotación (`Annotation_files/<vídeo>.txt`, cuyas dos primeras líneas son el frame de
  inicio y de fin de la caída; `0 0` = sin caída). Los vídeos sin anotación se omiten.
* Cita: I. Charfi, J. Mitéran, J. Dubois, M. Atri, R. Tourki. *Optimized
  spatio-temporal descriptors for real-time fall detection: comparison of SVM and
  Adaboost-based classification.* Journal of Electronic Imaging 22(4):041106, 2013.

Comprobar lo descargado: `python -m src.datasets --check`
