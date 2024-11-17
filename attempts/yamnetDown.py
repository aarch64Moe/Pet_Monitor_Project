import tensorflow as tf
import tensorflow_hub as hub

# Load YAMNet from TensorFlow Hub
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
yamnet_model.save('yamnet_model')
