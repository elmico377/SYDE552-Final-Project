import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa
# import librosa.display
import glob
import random
from chromagram import cnn_generate_t4, get_chroma_data_formatted
from stft import cnn_generate_t1, get_stft_data_formatted
from spectral_centroid import cnn_generate_t2, get_scentroid_data_formatted
from mel import cnn_generate_t3, get_mel_data_formatted
from plotter import plot_models, plot_model

def get_result(predict_output):
    if predict_output[0] > predict_output[1]:
        return 0
    else:
        return 1

def evaluate_get_f1_score(model, x_validation, y_validation):
    true_positives = 0
    true_negatives = 0
    false_negatives = 0
    false_positives = 0

    for i, current_test_case in enumerate(x_validation):
        test_case = np.expand_dims(current_test_case, axis=0)
        prediction = get_result(model.predict(test_case)[0])
        solution = y_validation[i]
        # print(prediction)
        if solution == 1:
            if solution == prediction:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if solution == prediction:
                true_negatives += 1
            else:
                false_positives += 1
    f1_score = true_positives / (true_positives + 0.5 * (false_positives + false_negatives))
    print("The f_1 score of this model is : " + str(f1_score))
    return f1_score


file_names = glob.glob('Audio-Files/Test-Words/Second-Trial/*')
random.shuffle(file_names)

MAX_SIZE = 30000

# Interactive mode to allow multiple graphs
plt.ion()

# current_approach = "chromagram"
# current_approach = "stft"
# current_approach = "scentroid"
# current_approach = "mel"
approach_list = ["chromagram"]#,"stft","scentroid","mel"]
for current_approach in approach_list:
    x_train_data = []
    y_train_labels = []

    x_test_data = []
    y_test_labels = []

    x_validation_data = []
    y_validation_labels = []
    
    for i, current_audio_file in enumerate(file_names):
        time_series, sample_rate = librosa.load(current_audio_file) # sr changes sample rate in the load command
        if MAX_SIZE - time_series.shape[0] < 0:
            print(current_audio_file)
            print(MAX_SIZE - time_series.shape[0])
        padding = np.zeros(MAX_SIZE - time_series.shape[0])
        padded_time_series = np.concatenate((time_series, padding))

        current_formatted_data = None
        if current_approach == "stft":
            current_formatted_data = get_stft_data_formatted(padded_time_series)
        elif current_approach == "scentroid":
            current_formatted_data = get_scentroid_data_formatted(padded_time_series, sample_rate)
        elif current_approach == "mel":
            current_formatted_data = get_mel_data_formatted(padded_time_series, sample_rate)
        else: #elif current_approach == "chromagram":
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
    if current_approach == "scentroid":
        x_train_data = np.expand_dims(np.array(x_train_data),2)
        y_train_labels = np.array(y_train_labels)
        x_test_data = np.expand_dims(np.array(x_test_data),2)
        y_test_labels = np.array(y_test_labels)
        x_validation_data = np.expand_dims(np.array(x_validation_data),2)
        y_validation_labels = np.array(y_validation_labels)
    else:
        x_train_data = np.expand_dims(np.array(x_train_data),3)
        y_train_labels = np.array(y_train_labels)
        x_test_data = np.expand_dims(np.array(x_test_data),3)
        y_test_labels = np.array(y_test_labels)
        x_validation_data = np.expand_dims(np.array(x_validation_data),3)
        y_validation_labels = np.array(y_validation_labels)

    plt.figure(figsize=(12,5))
    plt.subplot(1, 2, 1)
    plt.ylabel(current_approach + ' training accuracy')
    plt.xlabel('epochs')   
    plt.twinx()
    plt.ylabel('training loss (error)')
    plt.title('training')

    plt.subplot(1, 2, 2)
    plt.ylabel(current_approach + ' testing accuracy')
    plt.xlabel('epochs')
    plt.twinx()
    plt.ylabel('testing loss (error)')
    plt.title('testing')
    plt.tight_layout()
    plt.show()

    f_1_scores = []
    i = 0

    while i < 10:
        if current_approach == "stft":
            model = cnn_generate_t1(8, x_train_data, y_train_labels, x_test_data, y_test_labels)
        elif current_approach == "scentroid":
            model = cnn_generate_t2(6, x_train_data, y_train_labels, x_test_data, y_test_labels)
        elif current_approach == "mel":
            model = cnn_generate_t3(10, x_train_data, y_train_labels, x_test_data, y_test_labels)
        else: #elif current_approach == "chromagram":
            model = cnn_generate_t4(8, x_train_data, y_train_labels, x_test_data, y_test_labels)

        plt.subplot(1, 2, 1)
        plt.plot(model.history.history['accuracy'], c='k')
        plt.plot(model.history.history['loss'], c='b') 

        plt.subplot(1, 2, 2)
        plt.plot(model.history.history['val_accuracy'], c='k')
        plt.plot(model.history.history['val_loss'], c='b')

        f_1_score = evaluate_get_f1_score(model, x_validation_data, y_validation_labels)
        f_1_scores.append(f_1_score)
        i += 1

    print(f_1_scores)
    # score = model.evaluate(x_validation_data, y_validation_labels)

input("Press enter to finish")