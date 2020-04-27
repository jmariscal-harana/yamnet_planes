# imports 
import yamnet_original.params as params
import yamnet_original.features as features_lib
from yamnet_original.yamnet import _YAMNET_LAYER_DEFS

# TF / keras 
import tensorflow as tf
from tensorflow.keras import Model, layers


def yamnet(features):
  """Define the core YAMNet model in Keras."""
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


def yamnet_frames_model(feature_params):
  """Define the YAMNet waveform-to-class-scores model.

  Args:
    feature_params: An object with parameter fields to control the feature
    calculation.

  Returns:
    A model accepting (1, num_samples) waveform input and emitting a
    (num_patches, num_classes) matrix of class scores per time frame as
    well as a (num_spectrogram_frames, num_mel_bins) log-mel spectrogram feature
    matrix.
  """
  waveform = layers.Input(batch_shape=(1, None))
  # Store the intermediate spectrogram features to use in visualization.
  log_mel_spectrogram, _ = features_lib.waveform_to_log_mel_spectrogram(
    tf.squeeze(waveform, axis=0), feature_params)
  patches = features_lib.spectrogram_to_patches(log_mel_spectrogram, feature_params)
  net, predictions = yamnet(patches)
  frames_model = Model(name='yamnet_frames', 
                       inputs=waveform, outputs=[log_mel_spectrogram, patches, net, predictions])
  return frames_model
