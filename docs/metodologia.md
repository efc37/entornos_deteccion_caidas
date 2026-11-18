# Metodología y justificación · Trabajo 17: detección de caídas por visión

Este documento recoge las decisiones del proyecto y sirve de base para el póster.

## 1. Problema AAL

Las caídas son una de las principales causas de lesión y pérdida de autonomía en
personas mayores: la OMS estima unas 684 000 muertes anuales por caídas en el mundo.
Cuando la persona vive sola, el tiempo hasta recibir ayuda condiciona las
consecuencias. El servicio debe **detectar posibles caídas en una habitación y
diferenciarlas de actividades cotidianas** (sentarse, tumbarse, agacharse). También
debe **gestionar las falsas alarmas** con una confirmación y **proteger la privacidad**
de una cámara instalada en casa.

## 2. Estado del arte (resumen)

| Enfoque | Idea | Ventajas | Inconvenientes |
|---|---|---|---|
| Clásico (silueta, umbrales, SVM) | Sustracción de fondo; relación de aspecto, velocidad del centroide | Ligero, interpretable | Sensible al fondo, la iluminación y el punto de vista |
| CNN 2D por fotograma | Clasifica la postura en cada imagen | Rápido | Pierde la dinámica: confunde estar tumbado con haberse caído |
| CNN 3D / transformers de vídeo | Aprende de bloques de píxeles | Gran precisión dentro del dataset | Coste muy alto; usa la imagen completa (privacidad) |
| **Esqueleto + modelo temporal** | Pose (OpenPose, MediaPipe, YOLO-Pose) → LSTM/GRU, ST-GCN o transformer | Ligero, independiente del fondo, más privado | Depende de la calidad de la pose; oclusiones |

* La revisión sistemática de Gaya-Morey et al. (2024), sobre 151 trabajos, observa
  que 67 usan esqueletos y que OpenPose, AlphaPose y MediaPipe son los estimadores más
  comunes, seguidos de LSTM (29 trabajos) y GCN (9). También señala que las cifras no
  son comparables entre trabajos (protocolos distintos), que el 88 % de los datasets no
  incluye personas mayores y que la mitad de los trabajos no trata la privacidad.
* Trabajos recientes sobre esqueleto: ST-GCN para caídas (Keskes y Noumeir, 2021), rasgos
  de esqueleto con cámara RGB (Ramírez et al., 2021), transformers sobre pose (Raza et
  al., 2025), YOLOv11-Pose más un transformer compacto con validación en Le2i sin reajuste
  (Benabdennour et al., 2026), y un método ligero y trazable con YOLO12n y MediaPipe con
  reglas de anulación para reducir falsas alarmas, sin entrenamiento (Alam et al., 2026).
* Huecos que atacamos: (1) **generalización entre datasets** (pocos trabajos la
  evalúan), (2) **falsas alarmas medidas por evento y por hora**, no solo la exactitud
  por clip, y (3) **privacidad cuantificada** y no solo afirmada.

## 3. Propuesta de solución

**Objetivo:** servicio local de detección de caídas para una habitación que (a)
distinga la dinámica de una caída de actividades parecidas, (b) reduzca las falsas
alarmas mediante decisión temporal y confirmación con la persona, y (c) no almacene ni
transmita imágenes.

```
Cámara ─► YOLOv8n-Pose (local) ─► esqueleto 17 puntos ─► ventanas de 1,3 s ─► modelo temporal
   │            │ la imagen se descarta                        (GRU / ST-GCN)
   └────────────┘                                                   │ p(caída)
                                                                    ▼
  persistencia (3 ventanas) ─► ¿sigue en el suelo 2 s? ─► "¿Se encuentra bien?" ─► aviso a contacto
                                      │ no                     │ "estoy bien"
                                      ▼                        ▼
                                  descartada               cancelada
```

**Decisiones de diseño**
* *Esqueleto en el borde:* reduce el dato de ~27 MB/s (vídeo 640×480 a 30 fps sin comprimir)
  a ~6 KB/s (17 puntos × 3 valores × 4 bytes × 30 fps; cifras reales en la tabla E6), hace el
  modelo independiente del fondo y la iluminación y permite ejecutarlo en un mini PC.
* *Modelo temporal:* una caída es un **cambio rápido** de postura, mientras que estar
  tumbado es un estado. Una GRU unidireccional es causal y apta para streaming; la ST-GCN
  explota la estructura del cuerpo como grafo.
* *Normalización por ventana* (origen en la cadera, escala igual al torso) y
  *remuestreo a 25 fps*: necesarios para comparar datasets con distinta cámara y
  frecuencia de imagen.
* *Aumento de datos* (volteo, rotación, escala, pérdida de articulaciones, oclusión de
  piernas): pensado para mejorar la robustez ante el punto de vista y las oclusiones.
  Se evalúa como ablación (GRU con y sin aumento).
* *Gestión de la incertidumbre:* persistencia temporal, comprobación de inactividad y
  pregunta a la persona antes de avisar. Si la persona sale del campo de visión justo
  después de un posible caída, se pregunta igualmente.

## 4. Datos

| | URFD | Le2i |
|---|---|---|
| Procedencia | Univ. de Rzeszów (Polonia) | Le2i/ImViA, Univ. de Bourgogne (Francia) |
| Secuencias | 30 caídas + 40 ADL (cámara frontal) | 191 vídeos (los anotados) |
| Sensor | Kinect RGB 640×480, 30 fps (también profundidad e IMU) | RGB 320×240, 25 fps |
| Escenario | Laboratorio / habitación, 5 actores | 4 escenarios reales: Home, Coffee room, Office, Lecture room |
| Etiquetas | Por frame: no tumbado / cayendo / tumbado | Frame de inicio y fin de la caída |
| Licencia | CC BY-NC-SA 4.0 (no comercial) | Sin licencia explícita; uso académico citando el original |

**Selección y preparación.** Se usa solo la cámara frontal de URFD (comparable con
Le2i) y los vídeos de Le2i con anotación. Para cada fotograma: pose con YOLOv8n-Pose,
seguimiento de una persona, puntos con confianza menor de 0,3 tratados como no
visibles, interpolación de huecos cortos (hasta 12 frames), remuestreo a 25 fps y
ventanas de 32 frames con paso de 4. Etiqueta de ventana: **caída** si contiene al
menos 6 frames de la transición de caída; **ignorada** si solapa solo parcialmente
la caída o el tramo tumbado posterior; **no caída** en otro caso.

**Limitaciones.** Las caídas son simuladas por actores jóvenes; hay pocos sujetos
(URFD no publica a qué sujeto corresponde cada secuencia, por lo que se agrupa por
secuencia) y ángulos de cámara fijos; las grabaciones son cortas (las falsas alarmas
por hora son una extrapolación); y la etiqueta de Le2i marca el intervalo de la caída
de forma aproximada.

## 5. Modelos

| Tipo | Modelo | Entrada |
|---|---|---|
| Base sin entrenamiento | Reglas: tronco > 55°, descenso de cadera > 1,2 torsos/s y bbox más ancho que alto (2 de 3) | 25 features por ventana |
| Base clásico | Random Forest (300 árboles) y SVM RBF, con pesos balanceados | 25 features: ángulo del tronco y su velocidad, velocidad y aceleración vertical de cadera, centroide y cabeza, relación de aspecto, ratio de altura, velocidad articular, visibilidad |
| Propuesta | GRU (2 capas, 96 unidades) | Secuencia (32×17×5): x, y, confianza, dx, dy |
| Propuesta | ST-GCN ligera (3 bloques, grafo COCO) | Igual |

## 6. Protocolo experimental

* **E1 · URFD:** validación cruzada estratificada y agrupada por secuencia (5 particiones).
* **E2 · Le2i:** *leave-one-scene-out*; mide la generalización a una habitación nueva.
* **E3 · Entre datasets:** URFD → Le2i y Le2i → URFD, sin reajuste.
* **E4 · Por evento:** simulación en streaming (paso de 80 ms) con tres niveles de
  decisión: solo umbral, más persistencia, y más inactividad. Métricas: caídas detectadas,
  falsas alarmas por hora de actividad normal y latencia.
* **E5 · Campo de visión:** F1 por escena de Le2i y ante oclusiones simuladas
  (articulaciones perdidas al azar; piernas ocultas en el tramo final de la ventana).
* **E6 · Privacidad y coste:** bytes por segundo de cada representación, tiempo de
  pose y de inferencia.

Métricas por ventana: sensibilidad, especificidad, precisión, F1, exactitud
equilibrada y ROC-AUC (umbral 0,5). Se informa de la desviación entre particiones.

## 7. Integración en la casa inteligente

*Ejemplo de uso:* Carmen, de 82 años, vive sola. Por la noche resbala en el salón.
El servicio detecta el patrón de caída, comprueba durante 2 s que sigue en el suelo y
pregunta por el altavoz: «¿Se encuentra bien?». Si contesta «estoy bien», se cancela y
solo queda el registro local. Si pide ayuda o no responde en 30 s, se avisa a su hija
(contacto de confianza) con la hora del evento, sin imágenes.

*Errores e incertidumbre:*

| Situación | Respuesta del servicio |
|---|---|
| Falso positivo (se tumba en el sofá) | Se filtra por persistencia y dinámica; si llega a preguntar, la persona cancela |
| Falso negativo | Fallo más grave: el modelo se ajusta priorizando sensibilidad; botón de ayuda como respaldo |
| Persona fuera de plano o tapada | Aviso de campo de visión; si ocurre tras un posible caída, se pregunta |
| Baja confianza de la pose | Ventanas con más del 50 % de frames sin persona no generan alarma |

*Privacidad:* procesado local; en RAM solo hay unos segundos de esqueleto; en disco,
solo eventos. Eliminar la imagen **no garantiza el anonimato**: el esqueleto conserva
la forma de andar y las proporciones corporales, que pueden permitir reidentificar a
una persona. Por eso el esqueleto tampoco se almacena ni se envía. El servicio es
opcional y desactivable, y la persona decide los contactos.

## 8. Referencias

* Alam, E., Sufian, A., Dutta, P., Leo, M. (2026). A Pose Landmark-Based Lightweight Human Fall-Detection Framework Using Spatiotemporal Features. *Information* 17(10), 976. https://doi.org/10.3390/info17100976
* Alam, E., Sufian, A., Dutta, P., Leo, M. (2022). Vision-based human fall detection systems using deep learning: A review. *Computers in Biology and Medicine*. https://doi.org/10.1016/j.compbiomed.2022.105626
* Benkaci, A., Sliman, L., Dellys, H. N. (2024). Vision-based Human Fall Detection Systems: A Review. *Procedia Computer Science* 241, 203–211. https://doi.org/10.1016/j.procs.2024.08.028
* Benabdennour, A. et al. (2026). Real-Time Human Fall Detection From Video Using YOLOv11 With Pose Estimation. *IEEE Access* 14.
* Charfi, I., Mitéran, J., Dubois, J., Atri, M., Tourki, R. (2013). Optimized spatio-temporal descriptors for real-time fall detection. *J. Electronic Imaging* 22(4), 041106.
* Gaya-Morey, F. X., Manresa-Yee, C., Buades-Rubio, J. M. (2024). Deep learning for computer vision based activity recognition and fall detection of the elderly: a systematic review. *Applied Intelligence* 54, 8982–9007. https://doi.org/10.1007/s10489-024-05645-1
* Keskes, O., Noumeir, R. (2021). Vision-based fall detection using ST-GCN. *IEEE Access* 9, 28224–28236.
* Kwolek, B., Kepski, M. (2014). Human fall detection on embedded platform using depth maps and wireless accelerometer. *Comput. Methods Programs Biomed.* 117(3), 489–501.
* Ramírez, H., Velastín, S. A., et al. (2021). Fall detection and activity recognition using human skeleton features. *IEEE Access* 9, 33532–33542.
* Raza, A., Yousaf, M. H., Ahmad, W., Velastin, S. A., Viriri, S. (2025). Human fall detection using pose estimation: from traditional machine learning to vision transformers. *Eng. Appl. of AI* 143, 109809.
* Yan, S., Xiong, Y., Lin, D. (2018). Spatial temporal graph convolutional networks for skeleton-based action recognition. *AAAI*.
* Jocher, G. et al. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
* OMS. Falls (hoja informativa). https://www.who.int/news-room/fact-sheets/detail/falls
