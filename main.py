import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
# import librosa.display
import glob
import random
from chromagram import cnn_generate_t4, get_chroma_data_formatted
from stft import cnn_generate_t1, get_stft_data_formatted

file_names = glob.glob('Audio-Files/Test-Words/Initial-Trial/*')
random.shuffle(file_names)

x_train_data = []
y_train_labels = []

x_test_data = []
y_test_labels = []

x_validation_data = []
y_validation_labels = []

MAX_SIZE = 26000

# current_approach = "chromagram"
current_approach = "stft"

for i, current_audio_file in enumerate(file_names):
    time_series, sample_rate = librosa.load(current_audio_file) # sr changes sample rate in the load command
    padding = np.zeros(MAX_SIZE - time_series.shape[0])
    padded_time_series = np.concatenate((time_series, padding))

    current_formatted_data = None
    if current_approach == "stft":
        current_formatted_data = get_stft_data_formatted(padded_time_series)
    elif current_approach == "chromagram":
        current_formatted_data = get_chroma_data_formatted(padded_time_series, sample_rate) # Test with stft normal

    # print(current_formatted_data.shape)

    # Label is 1 for michael and 0 for other
    label = 0
    current_filename_chunks = current_audio_file.split('\\')
    current_filename = current_filename_chunks[len(current_filename_chunks)-1]
    if current_filename[0] == 'm':
        label = 1

    if i < len(file_names)/10.0: # First 10% is test data
        x_test_data.append(current_formatted_data)
        y_test_labels.append(label)
    elif i < len(file_names)/5.0: # Second 10% is validation data  
        x_validation_data.append(current_formatted_data)
        y_validation_labels.append(label)
    else: # Remaining data is test data
        x_train_data.append(current_formatted_data)
        y_train_labels.append(label)

x_train_data = np.expand_dims(np.array(x_train_data),3)
y_train_labels = np.array(y_train_labels)
x_test_data = np.expand_dims(np.array(x_test_data),3)
y_test_labels = np.array(y_test_labels)
x_validation_data = np.expand_dims(np.array(x_validation_data),3)
y_validation_labels = np.array(y_validation_labels)

if current_approach == "stft":
    model = cnn_generate_t1(10, x_train_data, y_train_labels, x_test_data, y_test_labels)
elif current_approach == "chromagram":
    model = cnn_generate_t4(10, x_train_data, y_train_labels, x_test_data, y_test_labels)
score = model.evaluate(x_validation_data, y_validation_labels)