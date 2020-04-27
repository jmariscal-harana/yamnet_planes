# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Feature computation for YAMNet"""

import numpy as np
import tensorflow as tf


def waveform_to_log_mel_spectrogram(waveform, params, print_on=0):
  """Compute log-mel spectrogram of a 1-D waveform."""
  with tf.name_scope('log_mel_features'): #context manager which adds "log_mel_features" to the name of each tensor
    # Convert waveform into spectrogram using a Short-Time Fourier Transform.
    # Note that tf.signal.stft() uses a periodic Hann window by default.
    stft_window_length = int(round(params.SAMPLE_RATE * params.STFT_WINDOW_SECONDS)) #STFT window length
    stft_hop_length = int(round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS)) #STFT hop length
    fft_length = 2 ** int(np.ceil(np.log(stft_window_length) / np.log(2.0))) #FFT size = smallest power of 2 enclosing stft_window_length
    num_spectrogram_bins = fft_length // 2 + 1

    if print_on:
      print("SPECTROGRAM PARAMETERS:")
      print("stft_window_length:",stft_window_length)
      print("stft_hop_length:",stft_hop_length)
      print("fft_length:",fft_length)
      print("num_spectrogram_bins:",num_spectrogram_bins,"\n")

    magnitude_spectrogram = tf.abs(tf.signal.stft(
      signals=waveform,
      frame_length=stft_window_length,
      frame_step=stft_hop_length,
      fft_length=fft_length)) # magnitude_spectrogram has shape [<# STFT frames>, num_spectrogram_bins]
    
    # Convert spectrogram into log-mel spectrogram.
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins=params.MEL_BANDS,
      num_spectrogram_bins=num_spectrogram_bins,
      sample_rate=params.SAMPLE_RATE,
      lower_edge_hertz=params.MEL_MIN_HZ,
      upper_edge_hertz=params.MEL_MAX_HZ)
    mel_spectrogram = tf.matmul(
      magnitude_spectrogram, 
      linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + params.LOG_OFFSET)
    # log_mel_spectrogram has shape [<# STFT frames>, MEL_BANDS]

    return log_mel_spectrogram, magnitude_spectrogram


def spectrogram_to_patches(spectrogram, params, print_on=0):
  """Break up any kind of spectrogram into a stack of fixed-size patches"""
  with tf.name_scope('feature_patches'):
    # Frame spectrogram (shape [<# STFT frames>, MEL_BANDS]) into patches
    # (the input examples).
    # Only complete frames are emitted, so if there is less than 
    # PATCH_WINDOW_SECONDS of waveform then nothing is emitted 
    # (to avoid this, zero-pad before processing).
    hop_length_samples = int(round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS))
    spectrogram_sr = int(round(params.SAMPLE_RATE / hop_length_samples))
    patch_window_length_samples = int(round(spectrogram_sr * params.PATCH_WINDOW_SECONDS))
    patch_hop_length_samples = int(round(spectrogram_sr * params.PATCH_HOP_SECONDS))
    
    if print_on:
      print("hop_length_samples:", hop_length_samples)
      print("spectrogram_sr:", spectrogram_sr)
      print("patch_window_length_samples:", patch_window_length_samples)
      print("params.PATCH_HOP_SECONDS:", params.PATCH_HOP_SECONDS)
      print("patch_hop_length_samples", patch_hop_length_samples)
    
    features = tf.signal.frame(
      signal=spectrogram,
      frame_length=patch_window_length_samples,
      frame_step=patch_hop_length_samples,
      axis=0)
    # features has shape [<# patches>, <# STFT frames in an patch>, MEL_BANDS]

    return features
