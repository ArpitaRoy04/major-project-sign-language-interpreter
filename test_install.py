import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from gtts import gTTS
import pyttsx3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  

print("All required libraries imported successfully.")