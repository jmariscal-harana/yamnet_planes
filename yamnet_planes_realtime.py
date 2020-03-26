import pyaudio, librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Decide what type of messages are displayed by TensorFlow (ERROR, WARN, INFO, DEBUG, FATAL)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


## TensorFlow memory allocation options:
# OPTION 1: "smart" allocation
#config=tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess=tf.Session(config=config) 

# OPTION 2: maximum memory allocation per session (0-1 = 0-100%)
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
sess=tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# OPTION 3: ???
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

# OPTION 4: ???
#Slim/embedded versions: (https://github.com/google-research/tf-slim)

print("tf version: ", tf.__version__)
print("tf.keras version: ", tf.keras.__version__)


## Add/append required paths
import os, sys

path_root = '/home/catec/Models/yamnet_planes/' #path to main folder
# path_root = input("Enter the path of your repository: ") # ask user for path_root
assert os.path.exists(path_root)
sys.path.append(path_root)

path_yamnet_original = path_root+'/yamnet_original/' #path to original yamnet files
assert os.path.exists(path_yamnet_original)
sys.path.append(path_yamnet_original)


## Modified YAMNet model for feature extraction
import modified_yamnet as yamnet_modified
import params

params.PATCH_HOP_SECONDS = 0.48 #low values: higher accuracy but higher computational cost

yamnet_features = yamnet_modified.yamnet_frames_model(params)
yamnet_features.load_weights(path_root+'yamnet.h5')


## Load yamnet_planes model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD, Adam

yamnet_planes = load_model(path_root+'top_model_v2.h5')
opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
yamnet_planes.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
yamnet_planes.summary()


## Function definitions
def run_models(waveform, 
               yamnet_features, 
               top_model, 
               strip_silence=True, 
               min_samples=16000):
    
    if strip_silence:
        waveform = remove_silence(waveform, top_db=10)
    
    if len(waveform) < min_samples:
        print("input too short after silence removal")
        return [-1] #this value will be used to discard unfit audios later
    
    _, _, dense_out, _ = yamnet_features.predict(np.reshape(waveform, [1, -1]), steps=1)
    
    # dense = (N, 1024)
    all_scores = []
    for patch in dense_out:
        scores = top_model.predict(np.expand_dims(patch,0)).squeeze()
        all_scores.append(scores)

    all_scores = np.mean(all_scores, axis=0)
    return all_scores


## Specify feature extraction parameters
import params
NUM_CLASSES = 2 #["not plane", "plane"]
categories = ["not plane", "plane"]
frame_len = int(params.SAMPLE_RATE * 1) # 1sec


## Start recording from microphone
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=params.SAMPLE_RATE,
                input=True,
                frames_per_buffer=frame_len)

cnt = 0
plt.ion()
while True:
    # Read data
    data = stream.read(frame_len, exception_on_overflow=False)

    # Change bit size
    waveform = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

    # Inference
    scores = run_models(waveform, yamnet_features, yamnet_planes, strip_silence=False, min_samples=8000)
    winner = categories[scores.argmax()]

    # Visualise spectrogram
#    _, spectrogram, _, _ = yamnet_features.predict(np.reshape(waveform, [1, -1]), steps=1)
#    plt.imshow(spectrogram.T, cmap='jet', aspect='auto', origin='lower')
#    plt.pause(0.001)
#    plt.show()

    print('Current event:\n' +
        "Best score: {}  label: {}".format(scores.max(), winner))

    # print idx
    print(cnt)
    cnt += 1

stream.stop_stream()
stream.close()
p.terminate()
