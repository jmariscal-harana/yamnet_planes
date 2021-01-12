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

def random_augment_wav(audio, sr):
    """
    Apply random augmentations to the sound:
        time stretch
        resample
        volume change
        minor random noise 
        TODO: probably a lot more augmentations you could use 

    Parameters
    ----------
    audio :
        Audio to be augmented.
    sr :
        Audio sampling rate.

    Returns
    -------
    audio_aug :
        Augmented audio.
    """
    audio_aug = audio
    
    if np.random.uniform() > 0.8: # random stretch 
        stretch = np.random.uniform(0.75, 1.5)
        audio_aug = librosa.effects.time_stretch(audio_aug, stretch)
    elif np.random.uniform() > 0.2: # random resample 
        new_sr = int(sr * np.random.uniform(0.9, 1.1))
        audio_aug = resampy.resample(audio_aug, sr, new_sr)    
    
    # Random volume
    volume = np.random.uniform(0.65, 1.2)
    audio_aug = audio_aug * volume
    
    # Random noise
    if np.random.uniform() > 0.5:
        NR = 0.001
        audio_aug += np.random.uniform(-NR, NR, size=audio_aug.shape)
    
    return audio_aug


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
    classes, 
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
    _samples = []
    _labels = []
    
    MIN_WAV_SIZE = int(DESIRED_SR * min_sample_seconds)
    MAX_WAV_SIZE = int(DESIRED_SR * max_sample_seconds)
    
    for label_idx, _class in enumerate(classes):
        label_dir = os.path.join(data_path, _class)
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


# https://github.com/tirthajyoti/Deep-learning-with-Python/blob/master/Notebooks/Custom-real-time-plots-with-callbacks.ipynb
import tensorflow as tf
from IPython.display import clear_output
import matplotlib.pyplot as plt
class TrainingPlot(tf.keras.callbacks.Callback):
    def __init__(self, path_save_file):
        self.path_save_file = path_save_file
    
    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and metrics
        self.losses = []
        self.acc = []
        self.f1score = []
        self.precision = []
        self.recall = []
        self.logs = []
    
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        """
        Calculates and plots Precision, Recall, F1 score
        """
        # Extract from the log
        # tp = logs.get('tp')
        # fp = logs.get('fp')
        # fn = logs.get('fn')
        loss = logs.get('loss')
        
        m = self.model
        # preds = m.predict(X_train)
        
        # Calculate
        # precision = tp/(tp+fp)
        # recall = tp/(tp+fn)
        # f1score = 2*(precision*recall)/(precision+recall)    
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(loss)
        # self.f1score.append(f1score)
        # self.precision.append(precision)
        # self.recall.append(recall)
        
        # Plots every 5th epoch
        if epoch > 0 and epoch%5==0:
            
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            
            # You can chose the style of your preference
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            # plt.figure(figsize=(10,3))
            # plt.title("Distribution of prediction probabilities at epoch no. {}".format(epoch), 
            #           fontsize=16)
            # plt.hist(preds, bins=50,edgecolor='k')
            
            plt.figure(figsize=(10,3))
            plt.title("Loss over epoch")
            plt.plot(N, self.losses)
            # fig, ax = plt.subplots(1,3, figsize=(12,4))
            # ax = ax.ravel()
            # ax[0].plot(N, self.precision, label = "Precision", c='red')
            # ax[1].plot(N, self.recall, label = "Recall", c='red')
            # ax[2].plot(N, self.f1score, label = "F1 score", c='red')
            # ax[0].set_title("Precision at Epoch No. {}".format(epoch))
            # ax[1].set_title("Recall at Epoch No. {}".format(epoch))
            # ax[2].set_title("F1-score at Epoch No. {}".format(epoch))
            # ax[0].set_xlabel("Epoch #")
            # ax[1].set_xlabel("Epoch #")
            # ax[2].set_xlabel("Epoch #")
            # ax[0].set_ylabel("Precision")
            # ax[1].set_ylabel("Recall")
            # ax[2].set_ylabel("F1 score")
            # ax[0].set_ylim(0,1)
            # ax[1].set_ylim(0,1)
            # ax[2].set_ylim(0,1)
            
            # Save figure to file
            plt.savefig(self.path_save_file)
            plt.savefig('./progress.png')

            # Plot figure (requires ipython)
            # plt.show()


import random

def balance_classes(features, labels):
    """
    Remove samples randomly to ensure class balance.

    Parameters
    ----------
    features :
        Samples to be randomised and removed.
    labels :
        Labels corresponding to each sample.

    Returns
    -------
    features :
        Samples (balanced).
    labels :
        Labels (balanced).
    """
    # Randomise sample/label order
    idxs_random = list(range(len(labels)))
    random.shuffle(idxs_random)
    features = [features[i] for i in idxs_random]
    labels = [labels[i] for i in idxs_random]

    # To ensure a balanced dataset, randomly delete [features,labels] from other classes to match number of features of least frequent class
    idx_labels, counts = np.unique(labels, return_counts=True)
    idx_locs_delete = []

    for idx in idx_labels:
        idx_locs = np.asarray(labels==idx).nonzero()[0]

        if len(idx_locs) > counts.min():
            idx_locs_delete.append(idx_locs[counts.min():])

    idx_locs_delete = idx_locs_delete[0].tolist()
    idx_locs_keep = list(set(range(len(features))) - set(idx_locs_delete))

    features = [features[i] for i in idx_locs_keep]
    labels = [labels[i] for i in idx_locs_keep]

    return features, labels


def extract_features(audio_list, yamnet_features):
    """
    Extract features for each audio the original yamnet model.
    
    Parameters
    ----------
    audio_list :
        Audios whose features will be extracted.
    yamnet_features :
        Original yamnet model used to extract audio features.

    Returns
    -------
    features_save :
        Extracted features.
    """
    features_save = []

    for audio in audio_list:
        # Extract features for current audio
        _, _, dense_out, _ = yamnet_features.predict(np.reshape(audio, [1, -1]), steps=1)
        
        # Add features to feature_save
        samples = []
        for patch in dense_out:
            samples.append(patch)

        features_save.append(samples)

    return features_save


def save_features(path_audio, sr, path_data_train, patch_hop_seconds_str, num_augmentations, class_idx, yamnet_features):
    # Read audio waveform
    audio = read_wav(path_audio, sr, use_rosa=True)

    # Check audio array dimension
    if audio.ndim > 2:
        raise Exception('Audio array can only be 1D or 2D.')
    elif audio.ndim == 2:
        # Downmix if multichannel
        audio = np.mean(audio, axis=1)

    # import matplotlib.pyplot as plt; plt.plot(audio); plt.savefig('/home/ups/Proyectos/vigia-sonido/Models/yamnet_planes/audio.png')

    # Avoid zero-padding within get_audio_embedding
    audio_min_dur = 1
    audio_curr_dur = max(audio.shape)/sr
    if audio_curr_dur <= audio_min_dur:
        print('Audio duration is < {} s for {}. Continue.'.format(audio_min_dur, os.path.basename(path_audio)))
        return

    # 1. Original audio
    audio_filename = ('_').join(path_audio.split(os.path.sep)[-2:]).split('.')[0]
    path_features = os.path.join(path_data_train, 'features', 'yamnet', audio_filename + '_features_' + patch_hop_seconds_str)
    path_labels = os.path.join(path_data_train, 'features', 'yamnet', audio_filename + '_labels_' +  patch_hop_seconds_str)

    path_features_aug = [path_features + '_00']
    path_labels_aug = [path_labels + '_00']

    if os.path.isfile(path_features_aug[-1] + '.npy'):
        audio_list, path_features_save, path_labels_save = [], [], []
    elif not os.path.isfile(path_features_aug[-1] + '.npy'):
        audio_list = [audio]
        path_features_save = path_features_aug.copy()
        path_labels_save = path_labels_aug.copy()

    # 2. Perform data augmentation on audio
    for idx_aug in range(num_augmentations[class_idx]):
        path_features_aug.append(path_features + '_{:02d}'.format(idx_aug+1))
        path_labels_aug.append(path_labels + '_{:02d}'.format(idx_aug+1))
        
        if not os.path.isfile(path_features_aug[-1] + '.npy'):
            print('Augmenting audio: {}'.format(idx_aug+1))
            audio_aug = random_augment_wav(audio, sr)
            audio_list.append(audio_aug)
            path_features_save.append(path_features_aug[-1])
            path_labels_save.append(path_labels_aug[-1])

    # Visualise data augmentations
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # for idx_row, row in enumerate(ax):
    #     for idx_col, col in enumerate(row):
    #         col.plot(audio_list[idx_row+idx_col+1])
    # plt.savefig('/home/ups/Proyectos/vigia-sonido/Models/yamnet_planes/audio_aug.png')

    if not audio_list:
        print('All features were previously extracted for {}. Continue.'.format(audio_filename))
        return

    features_save = extract_features(audio_list, yamnet_features)

    if len(features_save) != len(path_features_save):
        raise Exception('The number of extracted features is different from the number of features expected to be saved.')

    for features_tmp, path_features_tmp, path_labels_tmp in zip(features_save, path_features_save, path_labels_save):
        np.save(path_features_tmp, features_tmp)
        np.save(path_labels_tmp, np.full((len(features_tmp), 1), class_idx))    