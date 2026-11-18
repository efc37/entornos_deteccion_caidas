"""Configuración central del proyecto (rutas, hiperparámetros y umbrales).

Todos los scripts importan sus valores de aquí para que los experimentos sean
reproducibles y fáciles de modificar en un único sitio.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # datos originales (NO se versionan)
URFD_RAW = RAW_DIR / "urfd"
LE2I_RAW = RAW_DIR / "le2i"
SKELETON_DIR = DATA_DIR / "skeletons"   # esqueletos extraídos (.npz)
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

SEED = 42

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
URFD_BASE_URL = "https://fenix.ur.edu.pl/mkepski/ds/data"
URFD_N_FALLS = 30
URFD_N_ADLS = 40
URFD_FPS = 30.0          # Kinect / cámara frontal cam0
LE2I_FPS = 25.0

# ---------------------------------------------------------------------------
# Estimación de pose
# ---------------------------------------------------------------------------
POSE_MODEL = "yolov8n-pose.pt"   # versión nano: apta para dispositivos edge
POSE_CONF = 0.25                 # confianza mínima de la detección de persona
KP_CONF_MIN = 0.30               # por debajo, el keypoint se considera no visible

# Índices COCO-17 (formato de salida de YOLOv8-Pose)
KP_NAMES = [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
]
N_KP = 17
NOSE, L_SH, R_SH, L_HIP, R_HIP = 0, 5, 6, 11, 12
L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16
HEAD = [0, 1, 2, 3, 4]
LOWER_BODY = [11, 12, 13, 14, 15, 16]
# Pares izquierda/derecha (para el aumento por volteo horizontal)
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
# Aristas del esqueleto COCO (para la ST-GCN)
SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]

# ---------------------------------------------------------------------------
# Preprocesado y ventanas temporales
# ---------------------------------------------------------------------------
TARGET_FPS = 25.0        # ambos datasets se remuestrean a 25 fps
MAX_GAP_FRAMES = 12      # huecos (sin detección) más largos no se interpolan
WINDOW = 32              # ~1.3 s de contexto temporal
STRIDE_TRAIN = 4         # paso entre ventanas para entrenar/evaluar
STRIDE_STREAM = 2        # paso en la simulación del servicio (evaluación por evento)
MIN_FALL_FRAMES = 6      # nº mínimo de frames de "transición de caída" para etiquetar positiva
MAX_MISSING_FRAC = 0.5   # ventanas con más frames sin persona se marcan como "incierta"

# Etiquetas por frame
LBL_NORMAL, LBL_FALLING, LBL_LYING_AFTER = 0, 1, 2
# Etiquetas por ventana
WIN_NEG, WIN_POS, WIN_IGNORE = 0, 1, -1

# ---------------------------------------------------------------------------
# Baseline por reglas (sin entrenamiento)
# ---------------------------------------------------------------------------
RULE_TORSO_ANGLE = 55.0      # grados respecto a la vertical al final de la ventana
RULE_DESCENT_SPEED = 1.2     # torsos/s de velocidad descendente máxima de la cadera
RULE_ASPECT = 1.0            # alto/ancho del bbox del esqueleto al final de la ventana

# ---------------------------------------------------------------------------
# Modelos de aprendizaje
# ---------------------------------------------------------------------------
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                 n_jobs=-1, random_state=SEED)
SVM_PARAMS = dict(C=3.0, gamma="scale", class_weight="balanced", random_state=SEED)

DEEP = dict(
    hidden=96, layers=2, dropout=0.3,
    epochs=40, batch_size=128, lr=1e-3, weight_decay=1e-4,
    val_frac=0.15, patience=8,
)
AUG = dict(p_flip=0.5, max_rot_deg=15.0, scale=(0.85, 1.15), p_joint_drop=0.08,
           p_lower_occlusion=0.15)

# ---------------------------------------------------------------------------
# Decisión, postprocesado y servicio
# ---------------------------------------------------------------------------
PROB_THRESHOLD = 0.5
PERSIST_K = 3                 # nº de ventanas consecutivas por encima del umbral
INACTIVITY_S = 2.0            # segundos tras la caída para comprobar "en el suelo"
DOWN_FRAC = 0.6               # fracción de frames "en el suelo" para confirmar
REFRACTORY_S = 8.0            # tras una alarma no se generan otras durante este tiempo
EVENT_TOLERANCE_S = (1.0, 4.0)  # alarma válida en [inicio-1 s, fin+4 s]

CONFIRM_TIMEOUT_S = 30.0      # espera de respuesta a "¿Se encuentra bien?"
OUT_OF_VIEW_S = 10.0          # tiempo sin persona visible antes de avisar de campo de visión
