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

# Add/append required paths
import os, sys

path_root = '/home/anakin/Models/yamnet_planes/' #path to main folder
# path_root = input("Enter the path of your repository: ") # ask user for path_root
assert os.path.exists(path_root)
sys.path.append(path_root)

path_yamnet_original = path_root+'yamnet_original/' #path to original yamnet files
assert os.path.exists(path_yamnet_original)
# sys.path.append(path_yamnet_original)

# Load functions
import yamnet_functions

# Modified YAMNet model for feature extraction
import yamnet_original.params as params
import yamnet_modified as yamnet_modified

params.PATCH_HOP_SECONDS = 0.24 #low values: higher accuracy but higher computational cost
DESIRED_SR = params.SAMPLE_RATE # required by YAMNet

yamnet_features = yamnet_modified.yamnet_frames_model(params)
yamnet_features.load_weights(path_root+'yamnet.h5')

# Load model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam

yamnet_planes = load_model(path_root+'top_model.hdf5')
opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
yamnet_planes.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
yamnet_planes.summary()

# Specify class labels
class_labels = ["not plane", "plane"]

import yamnet_original.params as params
DESIRED_SR = params.SAMPLE_RATE # required by YAMNet

# Scores for a holdout folder
holdout_dir = "/home/anakin/Datasets/airplanes_v0/holdout_data/not_plane/"
arr = os.listdir(holdout_dir)

for fname in arr:
    print(fname)
    fname = holdout_dir+fname
    waveform = yamnet_functions.read_wav(fname, DESIRED_SR, use_rosa=1)

    # make file a bit longer by duplicating it 
    waveform = np.concatenate((waveform,waveform,waveform))
    scores = yamnet_functions.run_models(waveform, yamnet_features, yamnet_planes, strip_silence=False)
    winner_save = np.empty((0,2))
    if scores[0] != -1:
        winner = class_labels[scores.argmax()]
        print(" Best score: {}  label: {}".format(scores.max(), winner))
        # winner_save = np.append(winner_save,np.array([scores.max(),winner]),axis=0)
