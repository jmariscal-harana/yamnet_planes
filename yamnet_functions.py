# Function definitions
import pyaudio, librosa
import numpy as np
import soundfile as sf

# Convert input waveform to numpy 
def read_wav(
    wav_file, 
    output_sr,
    use_rosa=True):
    
    if use_rosa:
        waveform, sr = librosa.load(wav_file, sr=output_sr, dtype=np.float32)   # TODO: will np.float64 improve performance?
    else:
        wav_data, sr = sf.read(wav_file, dtype=np.int16)
        if wav_data.ndim > 1: 
            # (ns, 2)
            wav_data = wav_data.mean(1)
        if sr != output_sr:
            wav_data = resampy.resample(wav_data, sr, output_sr)
        waveform = wav_data / 32768.0
    
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
        NR = 0.001 # 0.001
        wav_data += np.random.uniform(-NR, NR, size=wav_data.shape)
    
    return wav_data


import glob, resampy
from tqdm import tqdm

def sample_count(
    data_path,
    params, 
    min_sample_seconds=1.0,
    max_sample_seconds=5.0,
    use_rosa=True,
    DESIRED_SR=16000):
    """Loads data from .wav files under data_path to count the number of samples from each class prior 
    to data augmentation.
    """
    # Get path to each folder within data_path
    label_dirs = get_top_dirs(data_path)
    
    MIN_WAV_SIZE = int(DESIRED_SR * min_sample_seconds)
    MAX_WAV_SIZE = int(DESIRED_SR * max_sample_seconds)
    
    sample_numbers = []

    print("Counting number of samples corresponding to each class\n")
    for label_idx, label_dir in enumerate(label_dirs):
        
        label_name = os.path.basename(label_dir)    # Get label name from current folder name
        wavs = glob.glob(os.path.join(label_dir, "*.wav"))  # Get all .wav file names within current folder
        sample_number = 0

        for wav_file in tqdm(wavs):
            waveform = read_wav(wav_file, DESIRED_SR, use_rosa=use_rosa)    # Read waveform

            if len(waveform) < MIN_WAV_SIZE:
                print("\nIgnoring audio shorter than {} seconds".format(min_sample_seconds))
                continue 

            if len(waveform) > MAX_WAV_SIZE:
                waveform = waveform[:MAX_WAV_SIZE]
                print("\nIgnoring audio data after {} seconds".format(max_sample_seconds))

            # Calculate number of samples
            audio_duration = len(waveform)/DESIRED_SR
            frame_duration = params.PATCH_WINDOW_SECONDS
            hop_duration = params.PATCH_HOP_SECONDS

            sample_number += 1 + np.floor((audio_duration - frame_duration)/hop_duration)

        sample_numbers.append(int(sample_number))

    for label_idx, label_dir in enumerate(label_dirs):
        label_name = os.path.basename(label_dir)

        print("'{}' samples: {}".format(label_name, sample_numbers[label_idx]))

    return sample_numbers


from scipy.io.wavfile import write

def data_augmentation(
    data_path, 
    yamnet_features,
    num_augmentations=[1,1],
    min_sample_seconds=1.0,
    max_sample_seconds=5.0,
    use_rosa=True,
    DESIRED_SR=16000):
    """Loads data from .wav files under data_path using subfolder names as labels,
    then runs them through yamnet_features to get feature vectors and returns them:
        X : [ np.array(1024) , ... ]
        Y : [ category_idx , ...]
    """
    print("Loading training data, number of augmentations = {}\n".format(num_augmentations))
    label_dirs = get_top_dirs(data_path)

    _samples = []
    _labels = []
    
    MIN_WAV_SIZE = int(DESIRED_SR * min_sample_seconds)
    MAX_WAV_SIZE = int(DESIRED_SR * max_sample_seconds)
    
    for label_idx, label_dir in enumerate(label_dirs):
        
        label_name = os.path.basename(label_dir)
        wavs = glob.glob(os.path.join(label_dir, "*.wav"))
        print("Loading {:<5}-> '{}'".format(label_idx, label_name))

        for wav_file in tqdm(wavs):
            waveform = read_wav(wav_file, DESIRED_SR, use_rosa=use_rosa)

            if len(waveform) < MIN_WAV_SIZE:
                print("\nIgnoring audio shorter than {} seconds".format(min_sample_seconds))
                continue 

            if len(waveform) > MAX_WAV_SIZE:
                waveform = waveform[:MAX_WAV_SIZE]
                print("\nIgnoring audio data after {} seconds".format(max_sample_seconds))

            for aug_idx in range((1 + num_augmentations[label_idx])):
                aug_wav = waveform.copy()
                
                if aug_idx > 0:
                    aug_wav = random_augment_wav(aug_wav, DESIRED_SR)

                    # # In case you want to listen to the augmented audios:
                    # if aug_idx == 1:
                    #     write('audio_original.wav', DESIRED_SR, waveform)
                    # write('audio_aug_'+str(aug_idx)+'.wav', DESIRED_SR, aug_wav)

                _, _, dense_out, _ = yamnet_features.predict(np.reshape(aug_wav, [1, -1]), steps=1)
                
                for patch in dense_out:
                    _samples.append(patch)
                    _labels.append(label_idx)

    # Calculate class percentages from labels
    for label_idx, label_dir in enumerate(label_dirs):
        label_name = os.path.basename(label_dir)
        label_name = "\'"+label_name+"\'"
        label_occurrences = _labels.count(label_idx)
        label_occurrences_percent = round(100*label_occurrences/len(_labels))

        print("{:<20} samples: {}%".format(label_name, label_occurrences_percent))

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
        print("The waveform is too short! Ignoring.")
        return [[-1, -1]] #this value will be used to discard this audio later
    
    _, _, dense_out, _ = yamnet_features.predict(np.reshape(waveform, [1, -1]), steps=1)
    
    # dense = (N, 1024)
    all_scores = []
    for patch in dense_out:
        scores = top_model.predict(np.expand_dims(patch,0)).squeeze()
        all_scores.append(scores)
        
    return all_scores


# Listen to audio function (workaround for VSCode)
# import scipy.io.wavfile, vlc, os

# def play_audio(
#     file_tmp,
#     sr,
#     waveform):

#     scipy.io.wavfile.write(file_tmp, sr, waveform)
#     for audio in [file_tmp]:
#         p = vlc.MediaPlayer(audio)
#         p.play()
#         print()
#     os.remove(audio)


# import sox

# def get_audio_durations(folder):
#     seconds_total = 0

#     files = [file for file in os.listdir(folder) if file.endswith('.wav')]
#     for file in files:
#         seconds = sox.file_info.duration(folder+file)
#         seconds_total += seconds

#     print('Folder: {}\nSeconds:  {:.2f}\nMinutes:    {:.2f}\nHours:      {:.2f}'.format(folder,seconds_total,seconds_total/60,seconds_total/3600))