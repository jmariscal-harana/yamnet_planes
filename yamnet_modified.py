# imports 
import yamnet_original.params as params
import yamnet_original.features as features
from yamnet_original.yamnet import _YAMNET_LAYER_DEFS

# TF / keras 
import tensorflow as tf
from tensorflow.keras import Model, layers


def yamnet(features):
  """Modified core YAMNet model (feature extraction only)

  Args:
    features: input image

  Returns:
    net: network configuration including the final global average pooling layer
    predictions (NOT USED): class scores per time frame matrix from original YAMNet
  """  
  net = layers.Reshape(
    (params.PATCH_FRAMES, params.PATCH_BANDS, 1),
    input_shape=(params.PATCH_FRAMES, params.PATCH_BANDS))(features)
  for (i, (layer_fun, kernel, stride, filters)) in enumerate(_YAMNET_LAYER_DEFS):
    net = layer_fun('layer{}'.format(i + 1), kernel, stride, filters)(net)
  net = layers.GlobalAveragePooling2D()(net)
  logits = layers.Dense(units=params.NUM_CLASSES, use_bias=True)(net)
  predictions = layers.Activation(
    name=params.EXAMPLE_PREDICTIONS_LAYER_NAME,
    activation=params.CLASSIFIER_ACTIVATION)(logits)
  return net, predictions


def yamnet_frames_model(params):
  """Define the YAMNet waveform-to-class-scores model.

  Args:
    params: parameters for feature calculation

  Returns:
    frames_model:
      Input:
        waveform: input waveform (1, num_samples)
      Output:
        log_mel_spectrogram: log-scaled mel spectrogram (num_spectrogram_frames, num_mel_bins)
        patches: image to be analysed
        net: network configuration including the final global average pooling layer
        predictions (NOT USED): class scores per time frame matrix from original YAMNet
  """
  waveform = layers.Input(batch_shape=(1, None))
  # Store the intermediate spectrogram features to use in visualization.
  log_mel_spectrogram, _ = features.waveform_to_log_mel_spectrogram(
    tf.squeeze(waveform, axis=0), params)
  patches = features.spectrogram_to_patches(log_mel_spectrogram, params)
  net, predictions = yamnet(patches)
  frames_model = Model(name='yamnet_frames', 
                       inputs=waveform, outputs=[log_mel_spectrogram, patches, net, predictions])
  return frames_model
