
      

##########################################
# GLOBAL IMPORTS
import pyaudio, librosa, os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import statistics 


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
path_data_csv = os.path.join(path_root, 'Datasets', dataset, 'data_csv')
path_yamnet_save = os.path.join(path_root, 'Models_saved/yamnet/')
path_yamnet_results = os.path.join(path_root, 'Results', dataset, 'yamnet')

assert os.path.exists(path_yamnet)
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
# 20210119_121442 urbansound+planes+some aircraft sounds from audioset
# 20210120_140743 urbansound+planes
# 20210121_155202 urbansound+planes+audioset
model_name = '20210120_140743'
scenario = 'und'

path_data_csv_file = os.path.join(path_data_csv, 'oran_microexterno_planes.csv')


##########################################
# LOAD MODELS
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam

# for params.PATCH_HOP_SECONDS in PATCH_HOP_SECONDS:
patch_hop_seconds_str = str(params.PATCH_HOP_SECONDS).replace('.','')

# 1. Load yamnet_features (feature extractor)
yamnet_features = yamnet_modified.yamnet_frames_model(params)
yamnet_features.load_weights(path_yamnet+'yamnet.h5')

print('Testing model {} for data {}\n'.format(model_name, os.path.basename(path_data_csv_file)))

path_yamnet_save_file = os.path.join(path_yamnet_save, model_name + '_yamnet_' + patch_hop_seconds_str + '_' + scenario + '.hdf5')

# 2. Load yamnet_planes (classifier)
yamnet_planes = load_model(path_yamnet_save_file)    
yamnet_planes.compile(optimizer=yamnet_planes.optimizer, loss=yamnet_planes.loss, metrics=yamnet_planes.metrics)
yamnet_planes.summary()

processed_samples = [0, 0]

detection_rate = []

df_data_csv = pd.read_csv(path_data_csv_file, header=None)
classes = np.unique(df_data_csv.iloc[:,1]).tolist()

# TODO:
classes = ['not_vehicle', classes[0]]

for class_idx, class_label in enumerate(classes):
    df_data_csv_class = df_data_csv[df_data_csv.iloc[:,1]==class_label]
    path_audios =  df_data_csv_class.iloc[:,0].tolist()

    if not path_audios:
        detection_rate.append([])
        continue

    predicted_class = []

    for path_audio in tqdm(path_audios):
        waveform, _ = librosa.load(path_audio, sr=DESIRED_SR, mono=True, dtype=np.float32)

        scores = yamnet_functions.run_models(waveform, yamnet_features, yamnet_planes, strip_silence=False)
        scores = np.array(scores)
        if scores[0][0] == -1:
            continue

        processed_samples[class_idx] += scores.shape[0]
        predicted_class.extend(scores.argmax(axis=1).tolist())

        # For each audio (from csv), print:
        print('{}: {:.2f}s, prediction: {}, confidence [min mean max] = [{:.2f} {:.2f} {:.2f}]'.format(
            os.path.basename(path_audio), len(waveform)/DESIRED_SR, classes[statistics.mode(scores.argmax(axis=1).tolist())], min(scores[:,class_idx]), np.mean(scores[:,class_idx]), max(scores[:,class_idx])))
        # plot time vs confidence
        plt.figure(figsize=(10,3))
        plt.title("Prediction score over time for class {}".format(class_label))
        plt.plot(scores[:,class_idx])
        plt.ylim((0, 1))
        plt.savefig(os.path.join(path_yamnet_results, 'score_plots', os.path.basename(path_audio) + '.png'))
        plt.close()
        
    detection_rate_current = sum(np.array(predicted_class) == class_idx) / len(predicted_class) * 100
    detection_rate.append(detection_rate_current)

    print('True positive rate for {} class (threshold = {:.2f}): {:4.2f}%'.format(class_label, 0.5, detection_rate[class_idx]))
    print('False negative rate for {} class (threshold = {:.2f}): {:4.2f}%'.format(class_label, 0.5, 100-detection_rate[class_idx]))