import tkinter as tk
from tkinter import Button, Label, StringVar
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import threading
import os
from my_functions import *

# Initialize the Tkinter GUI first
root = tk.Tk()
root.title("Sign Language Prediction")
root.geometry("500x300")

# Load and set background image
bg = tk.PhotoImage(file="back.png")  
background_label = tk.Label(root, image=bg)
background_label.place(relwidth=1, relheight=1)

# Initialize the TTS engine
engine = pyttsx3.init()

# Create the StringVar after the root window is initialized
predicted_word = StringVar()
predicted_word.set("Prediction will appear here")

# Load model and actions
PATH = 'data'
actions = np.array(os.listdir(PATH))
model = load_model('my_model.keras')

# Global variables for the predicted word and live feed
cap = cv2.VideoCapture(0)

# Function to process the video feed and make predictions
def process_video():
    global cap, predicted_word
    keypoints = []
    last_prediction = ""

    with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while cap.isOpened():
            _, frame = cap.read()
            if frame is None:
                break

            # Process frame for predictions
            results = image_process(frame, holistic)  # Custom function to process frame
            draw_landmarks(frame, results)  # Custom function to draw landmarks
            keypoints.append(keypoint_extraction(results))  # Extract keypoints

            if len(keypoints) == 10:
                keypoints_np = np.array(keypoints)
                prediction = model.predict(keypoints_np[np.newaxis, :, :])
                keypoints = []

                if np.amax(prediction) > 0.9:
                    new_prediction = actions[np.argmax(prediction)]
                    if new_prediction != last_prediction:
                        predicted_word.set(new_prediction)
                        last_prediction = new_prediction

            # Show the video feed in a separate OpenCV window
            cv2.imshow('Video Feed', frame)

            # Break on window close or `q` key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Function to handle TTS for the predicted word
def speak_prediction():
    word = predicted_word.get()
    if word and word != "Prediction will appear here":
        engine.say(word)
        engine.runAndWait()

# Function to start the video processing in a separate thread
def start_video_feed():
    threading.Thread(target=process_video, daemon=True).start()

# GUI Elements (Placed directly on root window)
Label(root, text="Real-Time Sign Language Prediction", font=("Arial", 16), bg="#d6f5ff").place(relx=0.5, rely=0.2, anchor="center")
Label(root, textvariable=predicted_word, font=("Arial", 14), fg="blue", bg="#d6f5ff").place(relx=0.5, rely=0.3, anchor="center")

Button(root, text="Start Video Feed", font=("Arial", 12), command=start_video_feed).place(relx=0.5, rely=0.4, anchor="center")
Button(root, text="Speak", font=("Arial", 12), command=speak_prediction).place(relx=0.5, rely=0.5, anchor="center")
Button(root, text="Exit", font=("Arial", 12), command=root.destroy).place(relx=0.5, rely=0.6, anchor="center")

# Run the GUI main loop
root.mainloop()
