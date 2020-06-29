# Function definitions
import pyaudio, librosa
import numpy as np

# Convert input waveform to numpy 
def read_wav(
    fname, 
    output_sr,
    use_rosa=True):
    
    if use_rosa:
        waveform, sr = librosa.load(fname, sr=output_sr, dtype=np.float32)
    else:
        wav_data, sr = sf.read(fname, dtype=np.int16)
        if wav_data.ndim > 1: 
            # (ns, 2)
            wav_data = wav_data.mean(1)
        if sr != output_sr:
            wav_data = resampy.resample(wav_data, sr, output_sr)
        waveform = wav_data / 32768.0
    
    #waveform = waveform.astype(np.float64) #will np.float64 improve performance?
    
    return waveform


# Loads sample into chunks of non-silence 
def remove_silence(
    waveform,
    top_db=15,
    min_chunk_size=2000,
    merge_chunks=True):

    splits = librosa.effects.split(waveform, top_db=top_db)
    waves = []
    for start, end in splits:
        if (end-start) < min_chunk_size:
            continue
        waves.append(waveform[start:end])
    
    if merge_chunks:
        out = None
        for c in waves:
            if out is None:
                out = c.copy()
            else:
                out = np.concatenate((out, c))
        waves = out
    
    return waves


import os 

def get_top_dirs(p):
    dirs = list(filter(lambda x : os.path.isdir( os.path.join(p, x) ), os.listdir(p)))
    return list(map(lambda x : os.path.join(p, x), dirs))

def random_augment_wav(
    wav_data,
    DESIRED_SR):

    # Apply random augmentations to the sound:
    # -time stretch, resample, volume change, minor noise 
    # -TODO: probably a lot more augmentations you could use 

    wav_data = wav_data.copy() 
    
    # random re-sample 
    if np.random.uniform() > 0.8:
        stretch = np.random.uniform(0.75, 1.5)
        wav_data = librosa.effects.time_stretch(wav_data, stretch)
    elif np.random.uniform() > 0.2:
        new_sr = int(DESIRED_SR * np.random.uniform(0.9, 1.1))
        wav_data = resampy.resample(wav_data, DESIRED_SR, new_sr)
    
    #librosa.effects.pitch_shift()
    
    # random volume
    volume = np.random.uniform(0.65, 1.2)
    wav_data = wav_data * volume
    
    # Random noise
    if np.random.uniform() > 0.5:
        NR = 0.001 # 0.1
        wav_data += np.random.uniform(-NR, NR, size=wav_data.shape)
    
    return wav_data


import glob, resampy
from tqdm import tqdm

def data_augmentation(
    data_path, 
    yamnet_features,
    num_augmentations=5,
    min_sample_seconds=1.0,
    max_sample_seconds=5.0,
    use_rosa=True,
    DESIRED_SR=16000):
    """Loads data from .wav files under data_path using subfolder names as labels,
    then runs them through yamnet_features to get feature vectors and returns them:
        X : [ np.array(1024) , ... ]
        Y : [ category_idx , ...]
    """
    print("Loading training data, number of augmentations = ", num_augmentations)    
    label_dirs = get_top_dirs(data_path)

    _samples = []
    _labels = []
    
    MIN_WAV_SIZE = int(DESIRED_SR * min_sample_seconds) #Should be at least 50% longer than PATCH_WINDOW_SECONDS
    MAX_WAV_SIZE = int(DESIRED_SR * max_sample_seconds)
    
    for label_idx, label_dir in enumerate(label_dirs):
        
        label_name = os.path.basename(label_dir)
        wavs = glob.glob(os.path.join(label_dir, "*.wav"))
        print(" Loading {:<5} '{:<40}'".format(label_idx, label_name))

        for wav_file in tqdm(wavs):
            
            # rosa seems very different?
            #for use_rosa in range(2):
            if True:
                #use_rosa = 1
                #use_rosa = np.random.uniform() > 0.5
                waveform = read_wav(wav_file, DESIRED_SR, use_rosa=use_rosa)

                if len(waveform) < MIN_WAV_SIZE:
                    continue 

                if len(waveform) > MAX_WAV_SIZE:
                    waveform = waveform[:MAX_WAV_SIZE]
                    print("\nIgnoring audio data after {} seconds".format(max_sample_seconds))

                for aug_idx in range(1 + num_augmentations):
                    
                    aug_wav = waveform.copy()
                    
                    if aug_idx > 0:
                        aug_wav = random_augment_wav(aug_wav, DESIRED_SR)

                    _, _, dense_out, _ = yamnet_features.predict(np.reshape(aug_wav, [1, -1]), steps=1)
                    
                    for patch in dense_out:
                        _samples.append(patch)
                        _labels.append(label_idx)

    for label_idx, label_dir in enumerate(label_dirs):
        label_name = os.path.basename(label_dir)
        label_occurrences = _labels.count(label_idx)

        print("Number of", label_name, "samples:", str(round(100*label_occurrences/len(_labels))) + "%")

    return _samples, _labels


def run_models(
    waveform, 
    yamnet_features, 
    top_model, 
    strip_silence=False, 
    min_samples=16000):
    
    if strip_silence:
        waveform = remove_silence(waveform, top_db=10)
    
    if len(waveform) < min_samples:
        print("input too short after silence removal")
        return [-1] #this value will be used to discard this audio later
    
    _, _, dense_out, _ = yamnet_features.predict(np.reshape(waveform, [1, -1]), steps=1)
    
    # dense = (N, 1024)
    all_scores = []
    for patch in dense_out:
        scores = top_model.predict(np.expand_dims(patch,0)).squeeze()
        all_scores.append(scores)
        
    all_scores = np.mean(all_scores, axis=0)
    return all_scores


# Listen to audio function (workaround for VSCode)
import scipy.io.wavfile, vlc, os

def play_audio(
    file_tmp,
    sr,
    waveform):

    scipy.io.wavfile.write(file_tmp, sr, waveform)
    for audio in [file_tmp]:
        p = vlc.MediaPlayer(audio)
        p.play()
        print()
    os.remove(audio)


# Calculate number of feature vectors which can be extracted from data to ensure class balance
# data_path = "/home/anakin/Datasets/airplanes_v0/training_data/"

# def class_balance(data_path):
    #open class and print out the class name
    #open each file within to extract their durations
    #feature_vectors = 1 + (duration - frame_length) / hop
    #feature_vectors_sum += feature_vectors
    #print(feature_vectors_sum)