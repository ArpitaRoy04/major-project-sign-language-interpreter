import cv2
import csv
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)

label = input("Enter label for this gesture (example: A, B, HELLO): ").strip().upper()
csv_file = f"{label}.csv"

write_header = not os.path.exists(csv_file)

cap = cv2.VideoCapture(0)
frame_index = 0

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)

    if write_header:
        header = ["label"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

    print("Show your hand in front of the camera.")
    print("Press q to stop recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect_for_video(mp_image, frame_index)
        frame_index += 1

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            row = [label]
            for lm in landmarks:
                row += [lm.x, lm.y, lm.z]
            writer.writerow(row)
            print("Saved one sample")

        cv2.putText(frame, f"Label: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()