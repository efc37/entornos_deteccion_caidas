import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Cargar el modelo pré-entrenado de estimación de pose
model = YOLO('yolov8n-pose.pt')  # Versión nano: ultra rápida y ligera

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_skeletons = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inferencia de pose
        results = model(frame, verbose=False)

        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                # Tomar keypoints de la persona detectada (17 puntos x 3 valores: x, y, conf)
                kps = result.keypoints.data[0].cpu().numpy()
                frame_skeletons.append(kps.flatten())  # Vetor plano de 51 elementos por frame
            else:
                # Si no se detecta persona en el frame, rellenar con ceros
                frame_skeletons.append(np.zeros(17 * 3))

    cap.release()
    return np.array(frame_skeletons)

if __name__ == '__main__':
    # Prueba rápida con un vídeo
    video_test = "data/raw/falls/fall-01-cam0.mp4"
    skeletons = process_video(video_test)
    print(f"Secuencia procesada correctamente: {skeletons.shape[0]} frames, {skeletons.shape[1]} datos por frame.")