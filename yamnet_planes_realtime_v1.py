import pyaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
#import keras

#import yamnet.params as params
#import yamnet.yamnet as yamnet_model
#yamnet = yamnet_model.yamnet_frames_model(params)
#yamnet.load_weights('yamnet/yamnet.h5')
#yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

##jmh
#from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import SGD, Adam
#from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

print("tf version: ", tf.__version__)
print("tf.keras version: ", tf.keras.__version__)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#
import os, sys
yamnet_base = './yamnet_base/'
assert os.path.exists(yamnet_base)
sys.path.append(yamnet_base)

import modified_yamnet as yamnet

# Load new model
from tensorflow.keras.models import load_model
model_new = load_model('top_model_v2.h5')
opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model_new.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_new.summary()

# Function definitions
def run_models(waveform, 
               yamnet_model, 
               top_model, 
               strip_silence=True, 
               min_samples=16000):
    
    if strip_silence:
        waveform = remove_silence(waveform, top_db=10)
    
    if len(waveform) < min_samples:
        print("input too short after silence removal")
        return [-1] #this value will be used to discard unfit audios later
    
    _, _, dense_out, _ = \
        yamnet_model.predict(np.reshape(waveform, [1, -1]), steps=1)
    
    # dense = (N, 1024)
    all_scores = []
    for patch in dense_out:
        scores = top_model.predict( np.expand_dims(patch,0)).squeeze()
        all_scores.append(scores)
        
    all_scores = np.mean(all_scores, axis=0)
    return all_scores

# Model parameters
import params
params.PATCH_HOP_SECONDS = 0.1  # 10 Hz scores frame rate.

DESIRED_SR = 16000
NUM_CLASSES = 2 # (plane, not plane)

categories = ["not plane", "plane"]

# Yamnet model for feature extraction
yamnet_model, dense_net = yamnet.yamnet_frames_model(params)
yamnet_model.load_weights('yamnet.h5')

##

frame_len = int(params.SAMPLE_RATE * 1) # 1sec

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=params.SAMPLE_RATE,
                input=True,
                frames_per_buffer=frame_len)

cnt = 0
plt.ion()
while True:
    # data read
    data = stream.read(frame_len, exception_on_overflow=False)

    # byte --> float
    frame_data = librosa.util.buf_to_float(data, n_bytes=2, dtype=np.int16)

    # model prediction
    #scores, melspec = yamnet.predict(np.reshape(frame_data, [1, -1]), steps=1)
    scores = run_models(frame_data, yamnet_model, model_new, strip_silence=False, min_samples=8000)
    winner = categories[scores.argmax()]

    #prediction = np.mean(scores, axis=0)

    # visualize input audio
#    plt.imshow(melspec.T, cmap='jet', aspect='auto', origin='lower')
#    plt.pause(0.001)
#    plt.show()

    #top5_i = np.argsort(prediction)[::-1][:5]

    # print result
#    print('Current event:\n' +
#          '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
#                    for i in top5_i))
    print('Current event:\n' +
        "Best score: {}  label: {}".format(scores.max(), winner))

    # print idx
    print(cnt)
    cnt += 1

stream.stop_stream()
stream.close()
p.terminate()
