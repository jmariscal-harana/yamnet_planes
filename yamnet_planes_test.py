import pyaudio, librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Decide what type of messages are displayed by TensorFlow (ERROR, WARN, INFO, DEBUG, FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable eager execution for TF1 compatibility
tf.compat.v1.disable_eager_execution()

# OPTION 2: maximum memory allocation per session (0-1 = 0-100%)
tf_ver = tf.__version__
if tf_ver[0] == "1":
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
    sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
elif tf_ver[0] == "2":
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.05)
    sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Add/append required paths
import os, sys

path_root = '/home/ups/Proyectos/Vigia_sonido/' #path to root folder
path_model = path_root+"Models/yamnet_planes/"
# path_model = input("Enter the path of your repository: ") # ask user for path_model
assert os.path.exists(path_model)
sys.path.append(path_model)

# Load functions
import yamnet_functions

# Modified YAMNet model for feature extraction
import yamnet_original.params as params
import yamnet_modified as yamnet_modified

params.PATCH_HOP_SECONDS = 0.096 # During testing, this should match the imposed inference time (0.48s ~= 2Hz)
DESIRED_SR = params.SAMPLE_RATE # required by YAMNet

yamnet_features = yamnet_modified.yamnet_frames_model(params)
yamnet_features.load_weights(path_model+'yamnet.h5')


# Load model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam

datetime = '20200825_011907' # TODO: extract from metadata
path_model_save = path_model+'saved_models/top_model_'+datetime+'.hdf5'

yamnet_planes = load_model(path_model_save)    
opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)    # TODO: extract from metadata
yamnet_planes.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # TODO: extract from metadata???
yamnet_planes.summary()

import yamnet_original.params as params
DESIRED_SR = params.SAMPLE_RATE # required by YAMNet

# Scores for testing folder
prediction_confidence = 0.5 # [0.5:1.0) where 0.5 is full confidence and ~1.0 is little confidence
class_labels = ["not_plane", "plane"]   # TODO: extract from metadata
path_data_test = path_root+'Datasets/airplanes_v3/holdout_data/'

detection_rate = []

for reference_class in range(len(class_labels)):
    path_data_test_class = path_data_test+class_labels[reference_class]+'/'
    arr = os.listdir(path_data_test_class)

    # path_data_test_class = path_model
    # arr = ['tractor.wav']

    predicted_class = []

    for fname in arr:
        print(fname)
        fname = path_data_test_class+fname
        waveform = yamnet_functions.read_wav(fname, DESIRED_SR, use_rosa=1)
        # make file a bit longer by duplicating it 
        # waveform = np.concatenate((waveform,waveform,waveform))
        scores = yamnet_functions.run_models(waveform, yamnet_features, yamnet_planes, strip_silence=False)
        scores = np.array(scores)

        prediction_loc, _ = np.where(scores>=prediction_confidence)
        scores = scores[prediction_loc, :]
        if prediction_loc.size == 0:
            print("Prediction not available for current prediction confidence: {}".format(prediction_confidence))
            continue

        if scores[0][0] != -1:
            predicted_class = np.append(predicted_class, scores.argmax(1))
            scores_mean = np.mean(scores, axis=0)
            winner_mean = class_labels[scores_mean.argmax()]
            print(" Best score: {}  label: {}".format(scores_mean.max(), winner_mean))

    detection_rate = np.append(detection_rate, sum(predicted_class==reference_class)/len(predicted_class)*100)
    print('True positive rate for {} class: {:4.2f}%'.format(class_labels[reference_class], detection_rate[reference_class]))
    print('False positive rate for {} class: {:4.2f}%'.format(class_labels[reference_class], 100-detection_rate[reference_class]))

# Write results file containing performance metrics
import pandas as pd

patch_hop_seconds_str = str(params.PATCH_HOP_SECONDS).replace('.','')
results_headers = ['path_model_save','patch_hop_seconds','reference_classes','prediction_confidence','TPR','FNR']
results_values = [path_model_save, patch_hop_seconds_str, class_labels, prediction_confidence, detection_rate, 100-detection_rate]
results_df = pd.DataFrame([results_values],columns=results_headers)
path_results = path_data_test+'test_'+os.path.basename(path_model_save)[:-5]+'_'+patch_hop_seconds_str+'_'+str(round(100*prediction_confidence))+'.csv'
results_df.to_csv(path_results,index=False)
