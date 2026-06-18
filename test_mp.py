import mediapipe as mp
print("MediaPipe version:", mp.__version__)
print("Has 'solutions':", hasattr(mp, 'solutions'))

if hasattr(mp, 'solutions'):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    print("mp.solutions.hands and mp.solutions.drawing_utils are available.")
else:
    print("Problem: mp.solutions is missing. Reinstall mediapipe.")