import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

dataset_path = r"E:\Major Project MCA\dataset"
output_csv = r"E:\Major Project MCA\isl_landmarks.csv"
model_path = r"E:\Major Project MCA\hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.HandLandmarker.create_from_options(options)

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["label"]
    for i in range(21):
        header += [f"x{i}", f"y{i}", f"z{i}"]
    writer.writerow(header)

    for root, dirs, files in os.walk(dataset_path):
        label = os.path.basename(root)
        for file in files:
            if os.path.splitext(file.lower())[1] not in image_exts:
                continue

            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if not result.hand_landmarks:
                continue

            landmarks = result.hand_landmarks[0]
            row = [label]
            for lm in landmarks:
                row += [lm.x, lm.y, lm.z]
            writer.writerow(row)

print("Saved:", output_csv)