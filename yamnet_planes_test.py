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


##########################################
# Add/append required paths
import os, sys

path_root = '/home/ups/Proyectos/Vigia_sonido/' #path to root folder
path_model = path_root+"Models/yamnet_planes/"
# path_model = input("Enter the path of your repository: ") # ask user for path_model
assert os.path.exists(path_model)
sys.path.append(path_model)


##########################################
# Load functions
import yamnet_functions

# Modified YAMNet model for feature extraction
import yamnet_original.params as params
import yamnet_modified as yamnet_modified


##########################################
# Specify test parameters
path_data_test = path_root+'Datasets/airplanes_v3/holdout_data/'
class_labels = ["not_plane", "plane"]   # TODO: extract from metadata

params.PATCH_HOP_SECONDS = 0.24 # During testing, this should match the imposed inference time (0.48s ~= 2Hz)
DESIRED_SR = params.SAMPLE_RATE # required by YAMNet

# Model name TODO: extract from metadata
# model_name = '20200824_234802' 	# Undersampling
# model_name = '20200823_025302'  # Data aug
# model_name = '20200823_210326'	# Data aug*2
# model_name = '20200825_011907'	# Hybrid

PATCH_HOP_SECONDS = [0.96, 0.48, 0.24, 0.096]
model_names = ['20200824_234802', '20200823_025302', '20200823_210326', '20200825_011907']


##########################################
# Load model and test
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam

test_mode = 2 # 1: proportional thresholding, 2: ROC thresholding
prediction_confidence = 0.5 # [0.5:1.0) where 0.5 is full confidence and ~1.0 is little confidence

# for params.PATCH_HOP_SECONDS in PATCH_HOP_SECONDS:

# Load YAMNet
yamnet_features = yamnet_modified.yamnet_frames_model(params)
yamnet_features.load_weights(path_model+'yamnet.h5')

for model_name in model_names:

    path_model_save = path_model+'saved_models/yamnet_'+model_name+'.hdf5'

    # Load yamnet_planes model
    yamnet_planes = load_model(path_model_save)    
    opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)    # TODO: extract from metadata
    yamnet_planes.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # TODO: extract from metadata???
    yamnet_planes.summary()

    processed_samples = [0, 0]

    # Method 1: apply equal threshold to all classes
    if test_mode == 1:        
        not_discarded = [0, 0]
        detection_rate = []

        for reference_class, class_label in enumerate(class_labels):
            path_data_test_class = path_data_test+class_label+'/'
            arr = os.listdir(path_data_test_class)

            predicted_class = []

            for fname in arr:
                print(fname)
                fname = path_data_test_class+fname
                waveform = yamnet_functions.read_wav(fname, DESIRED_SR, use_rosa=1)

                scores = yamnet_functions.run_models(waveform, yamnet_features, yamnet_planes, strip_silence=False)
                scores = np.array(scores)

                if scores[0][0] == -1:
                    continue
                        
                processed_samples[reference_class] += len(scores)
                prediction_loc, _ = np.where(scores>=prediction_confidence)
                scores = scores[prediction_loc, :]
                not_discarded[reference_class] += len(scores)

                if prediction_loc.size == 0:
                    print("Prediction not available for current prediction confidence: {}".format(prediction_confidence))
                    continue
            
                predicted_class = np.append(predicted_class, scores.argmax(1)).astype(int)
                scores_mean = np.mean(scores, axis=0)
                winner_mean = class_labels[scores_mean.argmax()]
                print(" Best score: {}  label: {}".format(scores_mean.max(), winner_mean))
                
            detection_rate = np.append(detection_rate, sum(predicted_class == reference_class) / len(predicted_class)*100)
            print('True positive rate for {} class: {:4.2f}%'.format(class_label, detection_rate[reference_class]))
            print('False positive rate for {} class: {:4.2f}%'.format(class_label, 100-detection_rate[reference_class]))


    # Method 2: apply moving threshold for ROC plots
    elif test_mode == 2:
        prediction_threshold_step = 0.01

        prediction_thresholds = np.arange(0, 1+prediction_threshold_step, prediction_threshold_step) # [0.0:1.0] where 0.5 is equal weighting for both classes
        prediction_thresholds = prediction_thresholds.reshape(len(prediction_thresholds), 1)
        detection_rate = np.empty((len(prediction_thresholds), 0), int)

        for reference_class, class_label in enumerate(class_labels):
            path_data_test_class = path_data_test+class_label+'/'
            arr = os.listdir(path_data_test_class)

            predicted_class = np.empty((len(prediction_thresholds), 0), int)

            for fname in arr:

                print(fname)
                fname = path_data_test_class+fname
                waveform = yamnet_functions.read_wav(fname, DESIRED_SR, use_rosa=1)

                scores = yamnet_functions.run_models(waveform, yamnet_features, yamnet_planes, strip_silence=False)
                scores = np.array(scores)
                if scores[0][0] == -1:
                    continue

                processed_samples[reference_class] += len(scores)
                scores = scores[:, 0]
                predicted_class = np.append(predicted_class, (scores <= prediction_thresholds), axis=1).astype(int)
                
            detection_rate_current = np.sum((predicted_class == reference_class), axis=1) / predicted_class.shape[1] * 100
            detection_rate = np.append(detection_rate, detection_rate_current.reshape(len(prediction_thresholds), 1), axis=1)

            for loc, prediction_threshold in enumerate(prediction_thresholds):
                print('True positive rate for {} class (threshold = {:.2f}): {:4.2f}%'.format(class_label, prediction_threshold[0], detection_rate[loc, reference_class]))
                print('False positive rate for {} class (threshold = {:.2f}): {:4.2f}%'.format(class_label, prediction_threshold[0], 100-detection_rate[loc, reference_class]))


    # Write results file containing performance metrics
    import pandas as pd

    patch_hop_seconds_str = str(params.PATCH_HOP_SECONDS).replace('.','')

    if test_mode == 1:
        results_headers = ['path_model_save','patch_hop_seconds','reference_classes','prediction_confidence','TP','FN','total_proc','not_discarded_by_threshold']
        results_values = [path_model_save, patch_hop_seconds_str, class_labels, prediction_confidence, detection_rate, 100-detection_rate, processed_samples, not_discarded]
        path_results = path_data_test+'test_'+os.path.basename(path_model_save)[:-5]+'_'+patch_hop_seconds_str+'_'+str(round(100*prediction_confidence))+'_'+str(not_discarded)+'.csv'
    elif test_mode == 2:
        results_headers = ['path_model_save','patch_hop_seconds','reference_classes','threshold','TP','FN','TN','FP','total_proc']
        TP = [int(x) for x in detection_rate[:,1]/100*processed_samples[1]]
        FN = [int(x) for x in (100-detection_rate[:,1])/100*processed_samples[1]]
        TN = [int(x) for x in detection_rate[:,0]/100*processed_samples[0]]
        FP = [int(x) for x in (100-detection_rate[:,0])/100*processed_samples[0]]
        results_values = [path_model_save, patch_hop_seconds_str, class_labels, prediction_thresholds.T[0], TP, FN, TN, FP, processed_samples]
        path_results = path_data_test+'test_'+os.path.basename(path_model_save)[:-5]+'_'+patch_hop_seconds_str+'_thresholding.csv'
        
        results_headers_ROC_PR = ['TP','FN','TN','FP']
        results_values_ROC_PR = np.array([TP, FN, TN, FP]).T
        path_results_ROC_PR = path_data_test+'test_'+os.path.basename(path_model_save)[:-5]+'_'+patch_hop_seconds_str+'_thresholding_ROC_PR.csv'
        results_df = pd.DataFrame(results_values_ROC_PR,columns=results_headers_ROC_PR)
        results_df.to_csv(path_results_ROC_PR,index=False)

    results_df = pd.DataFrame([results_values],columns=results_headers)
    results_df.to_csv(path_results,index=False)
