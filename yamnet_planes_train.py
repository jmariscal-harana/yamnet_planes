##########################################
# GLOBAL IMPORTS
import pyaudio, librosa, os, sys, datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from time import time
from tqdm import tqdm

# Decide what type of messages are displayed by TensorFlow (ERROR, WARN, INFO, DEBUG, FATAL)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable eager execution for TF1 compatibility
tf.compat.v1.disable_eager_execution()

# OPTION 2: maximum memory allocation per session (0-1 = 0-100%)
tf_ver = tf.__version__
if tf_ver[0] == "1":
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
elif tf_ver[0] == "2":
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=1.0)
    sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


##########################################
# PATHS
dataset = 'small_aircraft_v2'

path_root = '/home/ups/Proyectos/vigia-sonido/' #path to root folder
path_yamnet = os.path.join(path_root, 'Models/yamnet_planes/')
path_yamnet_original = os.path.join(path_yamnet, 'yamnet_original/') #path to original yamnet files
path_data_train = os.path.join(path_root, 'Datasets', dataset, 'training_data')
path_data_csv = os.path.join(path_root, 'Datasets', dataset, 'data_csv')
path_yamnet_save = os.path.join(path_root, 'Models_saved/yamnet/')

assert os.path.exists(path_yamnet)
assert os.path.exists(path_yamnet_original)
assert os.path.exists(path_data_train)
assert os.path.exists(path_yamnet_save)

sys.path.append(path_yamnet)


##########################################
# GLOBAL CONFIG
import yamnet_functions

# Modified YAMNet model for feature extraction
import yamnet_original.params as params
import yamnet_modified as yamnet_modified

params.PATCH_HOP_SECONDS = 0.096 #low values: higher accuracy but higher computational cost

patch_hop_seconds_str = str(params.PATCH_HOP_SECONDS).replace('.','')

yamnet_features = yamnet_modified.yamnet_frames_model(params)
yamnet_features.load_weights(path_yamnet+'yamnet.h5')


##########################################
# DATA ANALYSIS
# Count the number of original features for each class
min_sample_seconds=1  # Should be at least equal to params.PATCH_WINDOW_SECONDS
max_sample_seconds=1000.0

sample_numbers = [0]

# sample_numbers = yamnet_functions.sample_count(
#     path_data_train,
#     params, 
#     min_sample_seconds=min_sample_seconds,
#     max_sample_seconds=min_sample_seconds,
#     DESIRED_SR=params.SAMPLE_RATE)


##########################################
# DATA AUGMENTATION and FEATURE EXTRACTION
# Based on the number of original features decide how much data augmentation is required by each class
scenarios = ['und', 'aug1', 'aug2', 'hyb']
scenario = scenarios[0]
feature_extraction = True
feature_extraction_method = 1
path_data_csv_file = os.path.join(path_data_csv, 'audio_paths_all.csv')

if scenario == 'und':
    num_augmentations=[0,0]
elif scenario == 'aug1':
    num_augmentations=[0,12]
elif scenario == 'aug2':
    num_augmentations=[1,25]
elif scenario == 'hyb':
    num_augmentations=[0,7]
else:
    raise NameError(scenario+' is not a valid scenario')

# FEATURE EXTRACTION
if feature_extraction == True:
    # Option 1: old school
    if feature_extraction_method == 0:
        perform_augmentation = True
        classes = ['not_plane', 'plane']

        path_features = os.path.join(path_data_train, 'features', 'yamnet','yamnet_features_'+patch_hop_seconds_str+'_'+scenario)
        path_labels = os.path.join(path_data_train, 'features', 'yamnet','yamnet_labels_'+patch_hop_seconds_str+'_'+scenario)

        if perform_augmentation == True:
            features, labels = yamnet_functions.data_augmentation(
                path_data_train, 
                classes,
                yamnet_features,
                num_augmentations=num_augmentations,
                min_sample_seconds=min_sample_seconds,
                max_sample_seconds=max_sample_seconds,
                DESIRED_SR=params.SAMPLE_RATE)

            # Randomise sample/label order
            # TODO: use this function
            # features, labels = balance_classes(features, labels)

            import random

            idxs = list(range(len(labels)))
            random.shuffle(idxs)
            features = [features[i] for i in idxs]
            labels = [labels[i] for i in idxs]

            # NumPy arrays are more efficient, especially for large arrays
            features = np.array(features)
            labels = np.array(labels)

            # To ensure a balanced dataset, randomly delete [features,labels] from other classes to match number of features of least frequent class
            _, counts = np.unique(labels, return_counts=True)
            idx_locs_delete = []

            for idx in np.unique(labels):
                idx_loc = np.where(labels==idx)[0]

                if len(idx_loc) > counts.min():
                    idx_locs_delete = np.append(idx_locs_delete,idx_loc[counts.min()-1:-1])

            labels = np.delete(labels, idx_locs_delete.astype(int))
            features = np.delete(features, idx_locs_delete.astype(int), axis=0)

            # Save features and corresponding labels
            np.save(path_features, features)
            np.save(path_labels, labels)

    # Option 2: save individual features
    elif feature_extraction_method == 1:
        df_data_csv = pd.read_csv(path_data_csv_file, header=None)
        classes = np.unique(df_data_csv.iloc[:,1]).tolist()

        for class_idx, class_label in enumerate(classes):
            df_data_csv_class = df_data_csv[df_data_csv.iloc[:,1]==class_label]
            path_audios =  df_data_csv_class.iloc[:,0].tolist()
            
            # Save as audio name -> hop size -> augment cycle
            for path_audio in tqdm(path_audios):
                yamnet_functions.save_features(
                    path_audio,
                    params.SAMPLE_RATE,
                    min_sample_seconds,
                    path_data_train,
                    patch_hop_seconds_str,
                    num_augmentations,
                    class_idx,
                    yamnet_features)


##########################################
# FEATURE DATA LOADING
# Option 1: old school
if feature_extraction_method == 0:
    features = np.load(path_features + '.npy')
    labels = np.load(path_labels + '.npy')

    _, counts = np.unique(labels, return_counts=True)

    print("\nSample size: {} and type: {}".format(features.shape, features.dtype))
    print("Label size:  {}".format(labels.shape))
    print("Sample distribution per class (after balancing): {}\n".format(counts))

# Option 2: load individual features
elif feature_extraction_method == 1:
    print('Loading features and labels.\n')
    features = []
    labels = []

    # 1. read csv
    df_data_csv = pd.read_csv(path_data_csv_file, header=None)
    classes = np.unique(df_data_csv.iloc[:,1]).tolist()

    for class_idx, class_label in enumerate(classes):
        print('\nLoading class {}\n'.format(class_label))
        
        df_data_csv_class = df_data_csv[df_data_csv.iloc[:,1]==class_label]
        path_audios =  df_data_csv_class.iloc[:,0].tolist()

        # Load features for each audio and append features and labels
        for idx, path_audio in enumerate(tqdm(path_audios)):
            audio_filename = ('_').join(path_audio.split(os.path.sep)[-2:]).split('.')[0]
            path_features = os.path.join(path_data_train, 'features', 'yamnet', audio_filename + '_features_' + patch_hop_seconds_str)

            for idx_aug in range(num_augmentations[class_idx] + 1):
                path_features_tmp = path_features + '_{:02d}'.format(idx_aug) + '.npy'
                
                if os.path.isfile(path_features_tmp):
                    features_tmp = np.load(path_features_tmp)

                    for feature in features_tmp:
                        features.append(feature)
                        labels.append(class_idx)

                else:
                    print('WARNING: Cannot find file \n{}.'.format(path_features_tmp))


    ##########################################
    # FEATURE BALANCING
    print('Balancing class features.\n')
    _, counts = np.unique(labels, return_counts=True)
    print("Sample distribution per class (before balancing): {}\n".format(counts))

    features, labels = yamnet_functions.balance_classes(features, labels)

    _, counts = np.unique(labels, return_counts=True)
    print("Sample size: {} and type: {}".format(len(features), type(features[0])))
    print("Label size:  {}".format(len(labels)))
    print("Sample distribution per class (after balancing): {}\n".format(counts))

    features = [features] # Required when passing it to model.fit
    labels = [labels] # Required when passing it to model.fit


#TODO: temporary attempt to normalise features between [0 1]
# for idx, feature in enumerate(tqdm(features[0])):
#     features[0][idx] = (feature - min(feature)) / (max(feature) - min(feature))


##########################################
# CLASSIFIER
from tensorflow.keras import Model, layers

def yamnet_classifier(input_size=1024,
    num_hidden=[1024],
    num_classes=2):
    
    input_layer = layers.Input(shape=(input_size,)) #takes a vector of size (input_size,)
    dense_layer = layers.Dense(num_hidden[0], activation=None)(input_layer) #first classifier layer with (num_hidden[0]) neurons
    for idx_layer in range(1,len(num_hidden)):
        dense_layer = layers.Dense(num_hidden[idx_layer], activation=None)(dense_layer)
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
    num_hidden=num_hidden,
    num_classes=num_classes)

# Optimisation configuration
opt_type = 'SGD'
loss_type = 'sparse_categorical_crossentropy'
metrics = ['accuracy']

if opt_type == 'Adam':
    opt_conf = [0.001]
    opt = Adam(learning_rate=opt_conf[0])
elif opt_type == 'SGD':
    opt_conf = [0.001, 1e-6, 0.9, True]
    opt = SGD(lr=opt_conf[0], decay=opt_conf[1], momentum=opt_conf[2], nesterov=opt_conf[3])
else:
    raise NameError(opt_type+' is not a valid optimiser')

yamnet_planes.compile(optimizer=opt, loss=loss_type, metrics=metrics)


##########################################
# TRAINING
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

print('##########################################')
print('Training time, baby!\nScenario: {}'.format(scenario))
print('##########################################')

time_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
path_yamnet_save_model = os.path.join(path_yamnet_save, time_now+'_yamnet_'+patch_hop_seconds_str+'_'+scenario+'.hdf5')
path_yamnet_save_plot = os.path.join(path_yamnet_save, time_now+'_progress_'+patch_hop_seconds_str+'_'+scenario+'.png')

# Callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
save_best = ModelCheckpoint(path_yamnet_save_model, save_best_only=True, monitor='val_loss', mode='min')
plot_metrics = yamnet_functions.TrainingPlot(path_yamnet_save_plot) # real-time plots during training
# log_dir = "logs/{}".format(int(time()))
# tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

epochs = 10000
val_split = 0.1

# opt_conf = [0.001, 1e-6, 0.9, True]
# opt = SGD(lr=opt_conf[0], decay=opt_conf[1], momentum=opt_conf[2], nesterov=opt_conf[3])
# yamnet_planes.compile(optimizer=opt, loss=loss_type, metrics=metrics)

time_start = time()
history = yamnet_planes.fit(features, labels, epochs=epochs, validation_split=val_split, callbacks=[save_best,plot_metrics])
time_end = time()
time_train = round(time_end - time_start)
# results.append([decay, history.history['val_loss'], history.history['val_accuracy']])

# [print(result) for result in results]

train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']


##########################################
# METADATA
# Write metadata file containing hyperparameters for data augmentation and training
metadata_headers = ['path_data_train','patch_hop_seconds','features','aug','features_aug','classifier_conf','optimiser','optimiser_params','loss_function','path_model','epochs','val_split','train_loss','train_accuracy','val_loss','val_accuracy']
metadata_values = [path_data_train, params.PATCH_HOP_SECONDS, sample_numbers, num_augmentations, counts.tolist(), num_hidden, opt_type, opt_conf, loss_type, path_yamnet_save, epochs, val_split, train_loss, train_acc, val_loss, val_acc]
metadata_df = pd.DataFrame([metadata_values],columns=metadata_headers)
path_metadata = os.path.join(path_yamnet_save, time_now+'_metadata_'+patch_hop_seconds_str+'_'+scenario+'.csv')

metadata_df.to_csv(path_metadata,index=False)

print(f"{epochs} epochs in {round(time_train/3600, 3)} hours ({time_train/epochs} seconds per epoch)")
