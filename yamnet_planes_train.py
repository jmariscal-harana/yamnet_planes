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

path_root = '/home/anakin/' #path to root folder
path_model = path_root+"Models/yamnet_planes/"
# path_model = input("Enter the path of your repository: ") # ask user for path_model
assert os.path.exists(path_model)
sys.path.append(path_model)

path_yamnet_original = path_model+'yamnet_original/' #path to original yamnet files
assert os.path.exists(path_yamnet_original)

# Load functions
import yamnet_functions

# Modified YAMNet model for feature extraction
import yamnet_original.params as params
import yamnet_modified as yamnet_modified

params.PATCH_HOP_SECONDS = 0.24 #low values: higher accuracy but higher computational cost
DESIRED_SR=params.SAMPLE_RATE

yamnet_features = yamnet_modified.yamnet_frames_model(params)
yamnet_features.load_weights(path_model+'yamnet.h5')


# Data augmentation
path_data_train = path_root+"Datasets/airplanes_v0/training_data/"
# path_data_train = input("Enter the path of your training dataset: ") # ask user for path_data_train
num_augmentations=0

samples, labels = yamnet_functions.data_augmentation(
    path_data_train, 
    yamnet_features,
    num_augmentations=num_augmentations,
    min_sample_seconds=1.5,
    max_sample_seconds=1000.0,
    use_rosa=True,
    DESIRED_SR=DESIRED_SR)

# Randomise sample/label order
import random

idxs = list(range(len(labels)))
random.shuffle(idxs)

samples = [samples[i] for i in idxs]
labels = [labels[i] for i in idxs]

samples = np.array(samples)
labels = np.array(labels)

print(" Loaded samples: " , samples.shape, samples.dtype,  labels.shape)

# Classifier definition
from tensorflow.keras import Model, layers

def yamnet_classifier(input_size=1024,
    num_hidden=1024,
    num_classes=2):
    
    input_layer = layers.Input(shape=(input_size,)) #takes a vector of size (input_size,)
    dense_layer = layers.Dense(num_hidden, activation=None)(input_layer) #classifier layer with (num_hidden) neurons
    classifier_layer = layers.Dense(num_classes, activation='softmax')(dense_layer) #activation layer with (num_classes) neurons
    model = Model(inputs=input_layer, outputs=classifier_layer)
    return model

# Classifier parameters
from tensorflow.keras.optimizers import SGD, Adam

features_img_length = 1024
num_hidden = 1024
num_classes = 2

yamnet_planes = yamnet_classifier(
    input_size=features_img_length, 
    num_hidden=num_hidden,
    num_classes=num_classes)

# Optimisation configuration
#opt = Adam(learning_rate=0.001)
opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

yamnet_planes.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the classifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from time import time

# Callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
save_best = ModelCheckpoint('top_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
log_dir = "logs/{}".format(int(time()))
# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

epochs = 10

time_start = time()
history = yamnet_planes.fit(samples, labels, epochs=epochs, validation_split=0.1, callbacks=[save_best])
time_end = time()
time_train = round(time_end - time_start)

print(f"{epochs} epochs in {round(time_train/3600, 3)} hours ({time_train/epochs} seconds per epoch)")