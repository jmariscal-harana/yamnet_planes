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

"""Feature computation for YAMNet

Functions:
  waveform_to_log_mel_spectrogram: input a waveform and output a log-scaled mel spectrogram
  spectrogram_to_patches: input a log-scaled mel spectrogram and output the final image (features/patches) to be analysed"""
import numpy as np
import tensorflow as tf


def waveform_to_log_mel_spectrogram(waveform,params,print_on=0):
  """Compute log-scaled mel spectrogram of an input waveform"""

  with tf.name_scope('log_mel_features'): #context manager which adds "log_mel_features" to the name of each tensor
    # Calculate Short-Time Fourier Transform (STFT) parameters
    stft_frame_length = int(round(params.SAMPLE_RATE * params.STFT_WINDOW_SECONDS)) #STFT window length
    stft_frame_step = int(round(params.SAMPLE_RATE * params.STFT_HOP_SECONDS)) #STFT hop length
    fft_length = 2 ** int(np.ceil(np.log(stft_frame_length) / np.log(2.0))) #FFT length = smallest power of 2 enclosing stft_frame_length
    fft_frequency_bins = fft_length // 2 + 1 #number of frequency bins

    if print_on:
      print("SPECTROGRAM PARAMETERS:")
      print("stft_frame_length:",stft_frame_length)
      print("stft_frame_step:",stft_frame_step)
      print("fft_length:",fft_length)
      print("fft_frequency_bins:",fft_frequency_bins,"\n")

    # Convert waveform into spectrogram (complex values) using a STFT
    # NOTE: tf.signal.stft() uses a periodic Hann window by default
    spectrogram = tf.signal.stft(
      signals=waveform,
      frame_length=stft_frame_length,
      frame_step=stft_frame_step,
      fft_length=fft_length,
      pad_end=True) #[<# STFT frames>, fft_frequency_bins]

    # Calculate magnitude spectrogram
    magnitude_spectrogram = tf.abs(spectrogram) #[<# STFT frames>, fft_frequency_bins]
    
    # Calculate mel weight matrix
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
      num_mel_bins=params.MEL_BANDS,
      num_spectrogram_bins=fft_frequency_bins,
      sample_rate=params.SAMPLE_RATE,
      lower_edge_hertz=params.MEL_MIN_HZ,
      upper_edge_hertz=params.MEL_MAX_HZ)
    
    # Convert magnitude spectrogram into log-scaled mel spectrogram
    mel_spectrogram = tf.matmul(magnitude_spectrogram, linear_to_mel_weight_matrix)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + params.LOG_OFFSET) #[<# STFT frames>, MEL_BANDS]

    return log_mel_spectrogram, magnitude_spectrogram, spectrogram


def spectrogram_to_patches(spectrogram, params, print_on=0):
    """Break up any kind of spectrogram into multiple fixed-size images (features/patches)"""

    with tf.name_scope('feature_patches'):
      # Frame input spectrogram (e.g. log-scaled mel spectrogram) into multiple fixed-size image (features/patches)
      # Only complete images are emitted (if waveform < PATCH_WINDOW_SECONDS then nothing is emitted)
      # To avoid this, zero-pad with pad_end=TRUE
      feature_frame_length = int(round(params.PATCH_WINDOW_SECONDS/params.STFT_HOP_SECONDS))
      feature_frame_step = int(round(params.PATCH_HOP_SECONDS/params.STFT_HOP_SECONDS))
      
      if print_on:
        print("feature_frame_length:",feature_frame_length)
        print("feature_frame_step:",feature_frame_step,"\n")

      features = tf.signal.frame(
        signal=spectrogram,
        frame_length=feature_frame_length,
        frame_step=feature_frame_step,
        pad_end=False,
        pad_value=0,
        axis=0) # [<# images>, feature_frame_length, MEL_BANDS]

      return features