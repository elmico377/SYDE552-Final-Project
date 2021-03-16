import tensorflow.keras.datasets.mnist as mnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
import glob
import random

def cnn_generate_t4(num_epochs, x_train, y_train, x_test, y_test):
    input_shape = x_train.shape[1:]
    print(input_shape)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # tf.keras.layers.experimental.preprocessing.Resizing(12,30), # TODO: Play with resize layer change dimensions and interpolation style (bilinear trilinear)
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='softmax'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'] # in addition to the loss, also compute the categorization accuracy
                )
    model.fit(x_train, y_train, epochs=num_epochs, batch_size=10, validation_data=(x_test, y_test)) 
    return model

# def validate_model(model, x_data, y_labels):
#     num_correct = 0
#     predictions = model.predict(x_data)
#     print(predictions.shape)
#     for i, prediction in enumerate(predictions):
#         if prediction == y_labels[i]:
#             num_correct += 1
#     # for i, label in enumerate(y_labels):
#     #     prediction = model.predict(x_data[i])
#     #     if prediction == label:
#     #         num_correct += 1
#     print("num correct is: " + str(num_correct))
#     return num_correct/y_labels.shape[0]


file_names = glob.glob('Audio-Files/Test-Words/Initial-Trial/*')
random.shuffle(file_names)

x_train_data = []
y_train_labels = []

x_test_data = []
y_test_labels = []

x_validation_data = []
y_validation_labels = []

MAX_SIZE = 26000

for i, current_audio_file in enumerate(file_names):
    time_series, sample_rate = librosa.load(current_audio_file) # sr changes sample rate in the load command
    padding = np.zeros(MAX_SIZE - time_series.shape[0])
    padded_time_series = np.concatenate((time_series, padding))

    # ft_data = librosa.stft(time_series)
    hop_length = 512
    chromagram = librosa.feature.chroma_stft(padded_time_series, sr=sample_rate, hop_length=hop_length) # Test with stft normal

    # Label is 1 for michael and 0 for other
    label = 0
    current_filename_chunks = current_audio_file.split('\\')
    current_filename = current_filename_chunks[len(current_filename_chunks)-1]
    if current_filename[0] == 'm':
        label = 1

    if i < len(file_names)/10.0: # First 10% is test data
        x_test_data.append(chromagram)
        y_test_labels.append(label)
    elif i < len(file_names)/5.0: # Second 10% is validation data  
        x_validation_data.append(chromagram)
        y_validation_labels.append(label)
    else: # Remaining data is test data
        x_train_data.append(chromagram)
        y_train_labels.append(label)

x_train_data = np.expand_dims(np.array(x_train_data),3)
y_train_labels = np.array(y_train_labels)
x_test_data = np.expand_dims(np.array(x_test_data),3)
y_test_labels = np.array(y_test_labels)
x_validation_data = np.expand_dims(np.array(x_validation_data),3)
y_validation_labels = np.array(y_validation_labels)

model = cnn_generate_t4(10, x_train_data, y_train_labels, x_test_data, y_test_labels)
# accuracy = validate_model(model, x_validation_data, y_validation_labels)
score = model.evaluate(x_validation_data, y_validation_labels)


# plt.figure(figsize=(15, 5))
# librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
# plt.show()

#plt.figure()
#librosa.display.waveplot(time_series, sr = sample_rate)
# Option 1 - Short term Fourier Transform (Sometimes referred to as chromagram)
# f,t, Zxx = scipy.signal.stft(time_series,sample_rate) #nperseg is how many per segment

# reduced_stft_data = np.array_split(Zxx,3)
# reduced_f = np.array_split(f,3)

# from docs
# plt.pcolormesh(t, reduced_f[0], np.abs(reduced_stft_data[0]), shading='gouraud')
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()

# Option 2 - Spectral Centroid

# Option 3 - Mel-Frequency Cepstral Coefficients
