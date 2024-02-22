import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Replace 'D:/Animal Emotion Project/dogs dataset/dogs/train/dog' with the actual path to your dataset
dataset_path = 'D:/Animal Emotion Project/dogs dataset/dogs/train/dog'

# Function to extract features from audio files
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs

# Plot MFCCs for each emotion category
emotions = ['Happy', 'Sad', 'Neutral', 'Angry']

for emotion in emotions:
    emotion_folder = os.path.join(dataset_path, emotion)
    plt.figure(figsize=(10, 4))

    for file in os.listdir(emotion_folder)[:3]:  # Plot only the first 3 samples for each emotion
        if file.endswith(".wav"):
            file_path = os.path.join(emotion_folder, file)
            mfccs = extract_features(file_path)

            # Plot MFCCs
            plt.subplot(1, 3, os.listdir(emotion_folder).index(file) + 1)
            librosa.display.specshow(mfccs, x_axis='time')
            plt.colorbar()
            plt.title(f'{emotion} - {file}')
            plt.xlabel('Time')
            plt.ylabel('MFCC')

    plt.tight_layout()
    plt.show()
