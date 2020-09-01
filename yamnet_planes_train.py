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

path_yamnet_original = path_model+'yamnet_original/' #path to original yamnet files
assert os.path.exists(path_yamnet_original)


# Load functions
import yamnet_functions

# Modified YAMNet model for feature extraction
import yamnet_original.params as params
import yamnet_modified as yamnet_modified

params.PATCH_HOP_SECONDS = 0.096 #low values: higher accuracy but higher computational cost

yamnet_features = yamnet_modified.yamnet_frames_model(params)
yamnet_features.load_weights(path_model+'yamnet.h5')


# Count class samples, perform data augmentation, or load a specific augmented dataset
path_data_train = path_root+"Datasets/airplanes_v3/training_data/"
# path_data_train = input("Enter the path of your training dataset: ") # Ask user for path_data_train

# Count the number of original samples for each class
min_sample_seconds=params.PATCH_WINDOW_SECONDS  # Should be at least equal to params.PATCH_WINDOW_SECONDS
max_sample_seconds=1000.0

sample_numbers = [0]

# sample_numbers = yamnet_functions.sample_count(
#     path_data_train,
#     params, 
#     min_sample_seconds=min_sample_seconds,
#     max_sample_seconds=min_sample_seconds,
#     use_rosa=True,
#     DESIRED_SR=params.SAMPLE_RATE)

# Based on the number of original samples decide how much data augmentation is required by each class
num_augmentations=[0,7]
perform_augmentation = True

patch_hop_seconds_str = str(params.PATCH_HOP_SECONDS).replace('.','')
features_str = 'features_'+patch_hop_seconds_str+'_'+str(num_augmentations[0])+'_'+str(num_augmentations[1]) 
labels_str = 'labels_'+patch_hop_seconds_str+'_'+str(num_augmentations[0])+'_'+str(num_augmentations[1]) 

if perform_augmentation == True:
    samples, labels = yamnet_functions.data_augmentation(
        path_data_train, 
        yamnet_features,
        num_augmentations=num_augmentations,
        min_sample_seconds=min_sample_seconds,
        max_sample_seconds=max_sample_seconds,
        use_rosa=True,
        DESIRED_SR=params.SAMPLE_RATE)

    # Randomise sample/label order
    import random

    idxs = list(range(len(labels)))
    random.shuffle(idxs)
    samples = [samples[i] for i in idxs]
    labels = [labels[i] for i in idxs]

    # NumPy arrays are more efficient, especially for large arrays
    samples = np.array(samples)
    labels = np.array(labels)

    # TODO: To ensure a balanced dataset, randomly delete [samples,labels] from other classes to match number of samples of least frequent class
    _, counts = np.unique(labels, return_counts=True)
    idx_locs_delete = []

    for idx in np.unique(labels):
        idx_loc = np.where(labels==idx)[0]

        if len(idx_loc) > counts.min():
            idx_locs_delete = np.append(idx_locs_delete,idx_loc[counts.min()-1:-1])

    labels = np.delete(labels, idx_locs_delete.astype(int))
    samples = np.delete(samples, idx_locs_delete.astype(int), axis=0)

    # Save features and corresponding labels
    np.save(path_data_train+features_str, samples)
    np.save(path_data_train+labels_str, labels)

else:
    samples = np.load(path_data_train+features_str+'.npy')
    labels = np.load(path_data_train+labels_str+'.npy')

_, counts = np.unique(labels, return_counts=True)

print("\nSample size: {} and type: {}".format(samples.shape, samples.dtype))
print("Label size:  {}".format(labels.shape))
print("Sample distribution per class (after balancing): {}\n".format(counts))


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
num_hidden = [1024]
num_classes = 2

yamnet_planes = yamnet_classifier(
    input_size=features_img_length, 
    num_hidden=num_hidden[0],   # TODO: modify so that the length of num_hidden determines the number of hidden layers with for loop
    num_classes=num_classes)

# Optimisation configuration
opt_type = 'SGD'
loss_type = 'sparse_categorical_crossentropy'

if opt_type == 'Adam':
    opt_conf = [0.001]
    opt = Adam(learning_rate=opt_conf[0])
elif opt_type == 'SGD':
    opt_conf = [0.001, 1e-6, 0.9, True]
    opt = SGD(lr=opt_conf[0], decay=opt_conf[1], momentum=opt_conf[2], nesterov=opt_conf[3])
else:
    raise NameError(opt_type+' is not a valid optimiser')

yamnet_planes.compile(optimizer=opt, loss=loss_type, metrics=['accuracy'])

# Train the classifier
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from time import time
import datetime

# Callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
time_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
path_model_save = path_model+'saved_models/top_model_'+time_now+'.hdf5'
save_best = ModelCheckpoint(path_model_save, save_best_only=True, monitor='val_loss', mode='min')
# log_dir = "logs/{}".format(int(time()))
# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

epochs = 1000
val_split = 0.1

time_start = time()
history = yamnet_planes.fit(samples, labels, epochs=epochs, validation_split=val_split, callbacks=[save_best])
time_end = time()
time_train = round(time_end - time_start)

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']

# Write metadata file containing hyperparameters for data augmentation and training
import pandas as pd

metadata_headers = ['path_data_train','patch_hop_seconds','samples','aug','samples_aug','classifier_conf','optimiser','optimiser_params','loss_function','path_model','epochs','val_split','train_loss','train_accuracy','val_loss','val_accuracy']
# metadata_values = np.zeros((1,len(metadata_headers)), dtype=int)
metadata_values = [path_data_train, params.PATCH_HOP_SECONDS, sample_numbers, num_augmentations, counts.tolist(), num_hidden, opt_type, opt_conf, loss_type, path_model_save, epochs, val_split, train_loss, train_acc, val_loss, val_acc]
metadata_df = pd.DataFrame([metadata_values],columns=metadata_headers)
path_metadata = path_data_train+'metadata_'+patch_hop_seconds_str+'_'+str(num_augmentations[0])+'_'+str(num_augmentations[1])+'_'+time_now+'.csv'
metadata_df.to_csv(path_metadata,index=False)

print(f"{epochs} epochs in {round(time_train/3600, 3)} hours ({time_train/epochs} seconds per epoch)")