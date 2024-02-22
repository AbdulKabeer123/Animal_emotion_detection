import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Function to extract features from audio files
def extract_features(file_path):
    audio, _ = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=22050*2, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load the trained model
model = load_model('dog_emotion_model.h5')

# Replace 'path/to/test_audio_folder' with the actual path to your test audio files
test_audio_folder = 'D:\\Animal Emotion Project\\dogs dataset\\dogs\\test\\test'

# Process each test audio file
for file in os.listdir(test_audio_folder):
    if file.endswith(".wav"):
        file_path = os.path.join(test_audio_folder, file)

        # Extract features from the test audio file
        features = extract_features(file_path)

        # Reshape input data for Conv1D layer
        features = features.reshape(1, features.shape[0], 1)

        # Predict the emotion using the trained model
        prediction = model.predict(features)
        emotion_index = np.argmax(prediction)
        
        # Map the emotion index back to the actual emotion category
        emotions = ['Happy', 'Angry', 'Sad', 'Neutral']
        predicted_emotion = emotions[emotion_index]

        print(f"File: {file} - Predicted Emotion: {predicted_emotion}")
