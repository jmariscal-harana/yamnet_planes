##########################################
# GLOBAL IMPORTS
import pyaudio, librosa, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

# Disable eager execution for TF1 compatibility
tf.compat.v1.disable_eager_execution()

# Decide what type of messages are displayed by TensorFlow (ERROR, WARN, INFO, DEBUG, FATAL)
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# OPTION 2: maximum memory allocation per session (0-1 = 0-100%)
tf_ver = tf.__version__
if tf_ver[0] == "1":
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
elif tf_ver[0] == "2":
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.1)
    sess=tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


##########################################
# PATHS
dataset = 'small_aircraft_v2'

path_root = '/home/ups/Proyectos/vigia-sonido/' #path to root folder
path_yamnet = os.path.join(path_root, 'Models/yamnet_planes/')
path_data_test = os.path.join(path_root, 'Datasets', dataset, 'holdout_data')
path_data_csv = os.path.join(path_root, 'Datasets', dataset, 'data_csv')
path_yamnet_save = os.path.join(path_root, 'Models_saved/yamnet/')
path_yamnet_results = os.path.join(path_root, 'Results', dataset, 'yamnet')

assert os.path.exists(path_yamnet)
assert os.path.exists(path_data_test)
assert os.path.exists(path_data_csv)
assert os.path.exists(path_yamnet_save)
assert os.path.exists(path_yamnet_results)

sys.path.append(path_yamnet)


##########################################
# GLOBAL CONFIG
import yamnet_functions

# Modified YAMNet model for feature extraction
import yamnet_original.params as params
import yamnet_modified as yamnet_modified

# Specify test parameters
params.PATCH_HOP_SECONDS = 0.096 # During testing, this should match the imposed inference time (0.48s ~= 2Hz)
DESIRED_SR = params.SAMPLE_RATE # required by YAMNet


##########################################
# MODELS
# Model name TODO: extract from metadata
# model_name = '20200824_234802' 	# Undersampling
# model_name = '20200823_025302'  # Data aug
# model_name = '20200823_210326'	# Data aug*2
# model_name = '20200825_011907'	# Hybrid

# PATCH_HOP_SECONDS = [0.96, 0.48, 0.24, 0.096]
# models_to_load = ['20200824_234802', '20200823_025302', '20200823_210326', '20200825_011907'] # paper
# models_to_load = ['20201109_163947', '20201109_172806', '20201110_085452', '20201109_164143'] # 1000 epochs
models_to_load = ['20201117_103113', '20201117_180323', '20201117_130656', '20201117_103306'] # 3000 epochs
scenarios = ['und', 'aug1', 'aug2', 'hyb']

# 20210119_121442 urbansound+planes+some aircraft sounds from audioset
# 20210120_140743 urbansound+planes
# 20210121_155202 urbansound+planes+audioset


models_to_load = ['20210119_121442']
scenarios = ['und']

path_data_csv_file = os.path.join(path_data_csv, 'audio_paths_test.csv')


##########################################
# TEST CONFIG
test_mode = 2 # 1: proportional thresholding, 2: ROC thresholding
prediction_confidence = 0.5 # [0.5:1.0) where 0.5 is full confidence and ~1.0 is little confidence


##########################################
# LOAD MODELS
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam

# for params.PATCH_HOP_SECONDS in PATCH_HOP_SECONDS:
patch_hop_seconds_str = str(params.PATCH_HOP_SECONDS).replace('.','')

# 1. Load yamnet_features (feature extractor)
yamnet_features = yamnet_modified.yamnet_frames_model(params)
yamnet_features.load_weights(path_yamnet+'yamnet.h5')

for model_idx, model_name in enumerate(models_to_load):
    print('##########################################')
    print('Testing time, baby!\nScenario: {}'.format(scenarios[model_idx]))
    print('##########################################')

    path_yamnet_save_file = os.path.join(path_yamnet_save, model_name+'_yamnet_'+patch_hop_seconds_str+'_'+scenarios[model_idx]+'.hdf5')

    # 2. Load yamnet_planes (classifier)
    yamnet_planes = load_model(path_yamnet_save_file)    
    yamnet_planes.compile(optimizer=yamnet_planes.optimizer, loss=yamnet_planes.loss, metrics=yamnet_planes.metrics)
    yamnet_planes.summary()

    processed_samples = [0, 0]

    # Method 1: apply equal threshold to all classes
    if test_mode == 1:        
        not_discarded = [0, 0]
        detection_rate = []
        classes = ["not_plane", "plane"]   # TODO: extract from metadata

        for class_idx, class_label in enumerate(classes):
            path_class = os.path.join(path_data_test, class_label)
            audio_filenames = os.listdir(path_class)

            predicted_class = np.empty((len(prediction_thresholds), 0), int)

            for audio_filename in tqdm(audio_filenames):
                path_audio = os.path.join(path_class, audio_filename)
                waveform, _ = librosa.load(path_audio, sr=DESIRED_SR, mono=True, dtype=np.float32)

                scores = yamnet_functions.run_models(waveform, yamnet_features, yamnet_planes, strip_silence=False)
                scores = np.array(scores)

                if scores[0][0] == -1:
                    continue
                        
                processed_samples[class_idx] += len(scores)
                prediction_loc, _ = np.where(scores>=prediction_confidence)
                scores = scores[prediction_loc, :]
                not_discarded[class_idx] += len(scores)

                if prediction_loc.size == 0:
                    print("Prediction not available for current prediction confidence: {}".format(prediction_confidence))
                    continue
            
                predicted_class = np.append(predicted_class, scores.argmax(1)).astype(int)
                scores_mean = np.mean(scores, axis=0)
                winner_mean = classes[scores_mean.argmax()]
                print(" Best score: {}  label: {}".format(scores_mean.max(), winner_mean))
                
            detection_rate = np.append(detection_rate, sum(predicted_class == class_idx) / len(predicted_class)*100)
            print('True positive rate for {} class: {:4.2f}%'.format(class_label, detection_rate[class_idx]))
            print('False negative rate for {} class: {:4.2f}%'.format(class_label, 100-detection_rate[class_idx]))


    # Method 2: apply moving threshold for ROC plots
    elif test_mode == 2:
        prediction_threshold_step = 0.01

        prediction_thresholds = np.arange(0, 1+prediction_threshold_step, prediction_threshold_step) # [0.0:1.0] where 0.5 is equal weighting for both classes
        prediction_thresholds = prediction_thresholds.reshape(len(prediction_thresholds), 1)
        detection_rate = np.empty((len(prediction_thresholds), 0), int)

        # Option 1
        df_data_csv = pd.read_csv(path_data_csv_file, header=None)
        classes = np.unique(df_data_csv.iloc[:,1]).tolist()

        for class_idx, class_label in enumerate(classes):
            df_data_csv_class = df_data_csv[df_data_csv.iloc[:,1]==class_label]
            path_audios =  df_data_csv_class.iloc[:,0].tolist()

            predicted_class = np.empty((len(prediction_thresholds), 0), int)

            for path_audio in tqdm(path_audios):

        # Option 2
        # classes = ["not_plane", "plane"]   # TODO: extract from metadata

        # for class_idx, class_label in enumerate(classes):
        #     path_class = os.path.join(path_data_test, class_label)
        #     audio_filenames = os.listdir(path_class)

            # predicted_class = np.empty((len(prediction_thresholds), 0), int)

            # for audio_filename in tqdm(audio_filenames):
            #     # print(audio_filename)
            #     path_audio = os.path.join(path_class, audio_filename)

                waveform, _ = librosa.load(path_audio, sr=DESIRED_SR, mono=True, dtype=np.float32)

                scores = yamnet_functions.run_models(waveform, yamnet_features, yamnet_planes, strip_silence=False)
                scores = np.array(scores)
                if scores[0][0] == -1:
                    continue

                processed_samples[class_idx] += len(scores)
                scores = scores[:, 0]
                predicted_class = np.append(predicted_class, (scores <= prediction_thresholds), axis=1).astype(int)
                
            detection_rate_current = np.sum((predicted_class == class_idx), axis=1) / predicted_class.shape[1] * 100
            detection_rate = np.append(detection_rate, detection_rate_current.reshape(len(prediction_thresholds), 1), axis=1)

            for loc, prediction_threshold in enumerate(prediction_thresholds):
                print('True positive rate for {} class (threshold = {:.2f}): {:4.2f}%'.format(class_label, prediction_threshold[0], detection_rate[loc, class_idx]))
                print('False negative rate for {} class (threshold = {:.2f}): {:4.2f}%'.format(class_label, prediction_threshold[0], 100-detection_rate[loc, class_idx]))


    ##########################################
    # METADATA
    if test_mode == 1:
        results_headers = ['path_yamnet_save_file','patch_hop_seconds','counter_classes','prediction_confidence','TP','FN','total_proc','not_discarded_by_threshold']
        results_values = [path_yamnet_save_file, patch_hop_seconds_str, classes, prediction_confidence, detection_rate, 100-detection_rate, processed_samples, not_discarded]
        path_yamnet_results_csv_file = os.path.join(path_yamnet_results, model_name+'_'+patch_hop_seconds_str+'_'+scenarios[model_idx]+'_thresholding.csv')
    
    elif test_mode == 2:
        results_headers = ['path_yamnet_save_file','patch_hop_seconds','counter_classes','threshold','TP','FN','TN','FP','total_proc']
        TP = [int(x) for x in detection_rate[:,1]/100*processed_samples[1]]
        FN = [int(x) for x in (100-detection_rate[:,1])/100*processed_samples[1]]
        TN = [int(x) for x in detection_rate[:,0]/100*processed_samples[0]]
        FP = [int(x) for x in (100-detection_rate[:,0])/100*processed_samples[0]]
        results_values = [path_yamnet_save_file, patch_hop_seconds_str, classes, prediction_thresholds.T[0], TP, FN, TN, FP, processed_samples]
        path_yamnet_results_csv_file = os.path.join(path_yamnet_results, model_name+'_'+patch_hop_seconds_str+'_'+scenarios[model_idx]+'_thresholding_ROC_PR.csv')
        path_yamnet_results_pkl_file = os.path.join(path_yamnet_results, model_name+'_'+patch_hop_seconds_str+'_'+scenarios[model_idx]+'_thresholding_ROC_PR.pkl')
        
    results_df = pd.DataFrame([results_values],columns=results_headers)
    results_df.to_csv(path_yamnet_results_csv_file,index=False)
    results_df.to_pickle(path_yamnet_results_pkl_file)
