import numpy as np
import tensorflow as tf
import librosa

def cnn_generate_t2(num_epochs, x_train, y_train, x_test, y_test):
    input_shape = x_train.shape[1:]
    print(input_shape)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        # tf.keras.layers.experimental.preprocessing.Resizing(12,30), # TODO: Play with resize layer change dimensions and interpolation style (bilinear trilinear)
        tf.keras.layers.Conv1D(16, 3, activation='relu'),
        tf.keras.layers.Conv1D(32, 3, activation='relu'),
        tf.keras.layers.Conv1D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling1D(),
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
    model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test)) 
    return model

def get_scentroid_data_formatted(padded_time_series, sample_rate):
    # hop_length = 512
    # chromagram = librosa.feature.chroma_stft(padded_time_series, sr=sample_rate, hop_length=hop_length) # Test with stft normal
    scentroid_data = librosa.feature.spectral_centroid(padded_time_series, sr=sample_rate)[0]
    return scentroid_data