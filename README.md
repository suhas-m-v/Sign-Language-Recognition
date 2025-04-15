# Sign Language Recognition System 🤟

This is a Python-based Sign Language Recognition System that detects and recognizes hand gestures like **Hi**, **Bye**, **Thank You**, etc., using **MediaPipe**, **OpenCV**, and **TensorFlow/Keras**. The system supports real-time prediction and displays the detected word on-screen. Additionally, it can read the word aloud using text-to-speech.

## 🔥 Features

- Real-time hand tracking with MediaPipe  
- Gesture recognition using trained deep learning model  
- GUI built with Tkinter  
- Audio output for detected gestures (text-to-speech)  
- Extendable for more gestures  
- Lightweight (runs on CPU)

## 🛠 Tech Stack

- Python 3.10  
- OpenCV  
- MediaPipe  
- TensorFlow / Keras  
- Tkinter  
- pyttsx3 (for text-to-speech)

## 📸 Sample Gestures
Gesture | Meaning
✋ | Hi
👋 | Bye
🤲 | Thank You
👉 | You

## 🧠 Model Training
Dataset collected using MediaPipe landmarks

Each gesture class has 30 sequences of 30 frames

Trained on a simple LSTM model for sequence prediction

## 🔊 Text-to-Speech
Uses pyttsx3 to say the recognized gesture

Can be toggled on/off in the GUI

##💡 Future Improvements
Add more gesture classes (A-Z alphabet, full words)

Deploy as a web or mobile app

Improve accuracy with more training data

Integrate with voice assistant

## 🤝 Contributions
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
