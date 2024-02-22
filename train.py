


##########################

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D

# Function to extract features from audio files
def extract_features(file_path):
    audio, _ = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=22050*2, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load and preprocess the dataset
def load_and_preprocess_dataset(dataset_path):
    data = []
    labels = []

    for emotion in os.listdir(dataset_path):
        emotion_folder = os.path.join(dataset_path, emotion)
        
        for file in os.listdir(emotion_folder):
            if file.endswith(".wav"):
                file_path = os.path.join(emotion_folder, file)
                features = extract_features(file_path)
                data.append(features)
                labels.append(emotion)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    labels = to_categorical(labels)

    return np.array(data), labels

# Replace 'path/to/train_dog_audio' with the actual path to your training dataset
train_dataset_path = 'D:\\Animal Emotion Project\\dogs dataset\\dogs\\train\\dog'
X_train, y_train = load_and_preprocess_dataset(train_dataset_path)

# Build the CNN model
model = Sequential()
model.add(Conv1D(128, 5, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape input data for Conv1D layer
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Save the trained model for future use
model.save('dog_emotion_model.h5')
