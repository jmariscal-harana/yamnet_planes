# yamnet_planes: small aircraft detection from sound
This model is based on Google's YAMNet pretrained deep network, and it has been modified to detect small aircraft sounds in real time. The original repository can be found at: https://github.com/tensorflow/models/tree/master/research/audioset/yamnet.


## Folder description (23/06/2020)
- **yamnet_original/**: contains the following scripts from YAMNet:
  - `yamnet.py`: original model definition
  - `params.py`: hyperparameter list
  - `features.py`: audio feature extraction functions
- `yamnet_modified.py`: modified version of `yamnet.py` which outputs the network up to and including the 'global average pooling' layer
- `yamnet_functions.py`: python functions required for data processing
- `yamnet_planes_realtime.py`: real-time inference from microphone feed


## Install Python3 dependencies 
*pip3 install -U numpy librosa pyaudio matplotlib*


## Inference
The model can be currently detect small planes without the need to train it.


### Setting up 'yamnet_planes'
The only parameter in `yamnet_original/params.py` which should be modified during inference is PATCH_HOP_SECONDS: a smaller hop should give you more patches from the same clip and possibly better performance at a larger computational cost.


### Running 'yamnet_planes'
The model can be used by running `python3 yamnet_planes_realtime.py`. It will take up to 1 minute to load. Once it loads, if you have connected a microphone to your device, it will start detecting planes!

The code will report, every PATCH_HOP_SECONDS seconds, the detected class ('plane' or 'not plane') and the average prediction confidence (from 0 to 1) for the current time interval.


### Input: Audio Features
See `features.py`. YAMNet was trained with audio features computed as follows:

* All audio is resampled to 16 kHz mono.
* A spectrogram is computed using magnitudes of the Short-Time Fourier Transform with a window size of 25 ms, a window hop of 10 ms, and a periodic Hann window.
* A mel spectrogram is computed by mapping the spectrogram to 64 mel bins covering the range 125-7500 Hz.
* A log-scaled mel spectrogram is computed by applying log(mel-spectrogram + 0.001) where the offset is used to avoid taking a logarithm of zero.
* These features are then framed into XX%-overlapping patches (images) of 0.96 seconds, where each patch covers 64 mel bands and 96 frames of 10 ms each.

These 96x64 patches are then fed into the Mobilenet_v1 model to yield a 3x2 array of activations for 1024 kernels at the top of the convolution.
These are averaged to give a 1024-dimension embedding (feature vector), then put through a single logistic layer to get the 2 per-class output scores corresponding to the 0.96 second input waveform segment.


## Training
If you want to train the model, try `yamnet_planes_train` (jupyter notebook or python script) with your own data.


## Performance
TBC ...


## Contact information
This repository is maintained by [Jorge Mariscal Harana](https://github.com/jmariscal-harana).
