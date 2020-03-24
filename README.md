# yamnet_planes: small airplane detection from sound
This model is based on Google's YAMNet pretrained deep network, and it has been modified to detect small airplane sounds in real time. The original repository can be found at https://github.com/tensorflow/models/tree/master/research/audioset/yamnet.


## File and folder description (24/03/2020)
- **yamnet_original/**: contains the following scripts from YAMNet:
  - `yamnet.py`: original model definition
  - `params.py`: hyperparameter list
  - `features.py`: audio feature extraction functions
- `modified_yamnet.py`: modified version of `yamnet.py` which outputs the network up to and including the 'global average pooling' layer
- `yamnet_planes_realtime_v2.py`: real-time inference from microphone feed


## Install Python3 dependencies 
*pip3 install -U numpy librosa pyaudio matplotlib*


## Inference
The model can be currently detect small planes without the need to train it.


### Setting up 'yamnet_planes'
The only parameter in `yamnet_original/params.py` which should be modified during inference is PATCH_HOP_SECONDS: a smaller hop should give you more patches from the same clip and possibly better performance at a larger computational cost.


### Running 'yamnet_planes'
The model can be used by running `python3 yamnet_planes_realtime_v2.py`. It will take up to 1 minute to load. Once it loads, if you have connected a microphone to your device, it will start detecting planes!

The code will report, every ?? seconds, the detected class ('plane' or not 'plane') and the average confidence averaged over all the frames for the current time interval.


### Input: Audio Features
See `features.py`. YAMNet was trained with audio features computed as follows:

* All audio is resampled to 16 kHz mono.
* A spectrogram is computed using magnitudes of the Short-Time Fourier Transform
  with a window size of 25 ms, a window hop of 10 ms, and a periodic Hann
  window.
* A mel spectrogram is computed by mapping the spectrogram to 64 mel bins
  covering the range 125-7500 Hz.
* A stabilized log mel spectrogram is computed by applying
  log(mel-spectrum + 0.001) where the offset is used to avoid taking a logarithm
  of zero.
* These features are then framed into 50%-overlapping examples of 0.96 seconds,
  where each example covers 64 mel bands and 96 frames of 10 ms each.

These 96x64 patches are then fed into the Mobilenet_v1 model to yield a 3x2
array of activations for 1024 kernels at the top of the convolution.  These are
averaged to give a 1024-dimension embedding, then put through a single logistic
layer to get the 2 per-class output scores corresponding to the 960 ms input
waveform segment. (Because of the window framing, you need at least 975 ms of
input waveform to get the first frame of output scores.)


## Training
If you want to improve the model...


## Performance
...


## Contact information
This repository is maintained by [Jorge Mariscal Harana](https://github.com/jmariscal-harana).
