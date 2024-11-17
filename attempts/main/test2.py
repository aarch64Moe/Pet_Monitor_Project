import numpy as np
import sounddevice as sd
import tensorflow as tf
from scipy import signal
from collections import Counter

# Load your classifier model (replace with actual model path)
classifier_model_path = "/home/moe/PycharmProjects/Download_preparev2/fine_tuned_classifier"
classifier_model = tf.keras.models.load_model(classifier_model_path)

# Load the YAMNet model (replace with actual model path)
yamnet_model_path = "/home/moe/PycharmProjects/Download_preparev2/yamnet_model"
yamnet_model = tf.saved_model.load(yamnet_model_path)

# Set audio parameters
SAMPLE_RATE = 16000  # Match the model's sample rate
DURATION = 3  # 3 seconds buffer for prediction
BUFFER_SIZE = SAMPLE_RATE * DURATION

# Define your class names (Ensure these match your model's output classes)
class_names = ["Angry", "Attention", "Fighting", "Happy", "HuntingMind", "Kittins", "Mating", "Purring"]

# Initialize audio buffer globally
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
predictions = []  # Store predictions globally


def extract_yamnet_embeddings(audio_buffer):
    # Ensure audio_buffer is in the correct shape (flatten to 1D)
    audio_tensor = tf.convert_to_tensor(audio_buffer, dtype=tf.float32)
    audio_tensor = tf.reshape(audio_tensor, [-1])  # Flatten to 1D tensor
    yamnet_output = yamnet_model(audio_tensor)  # Get YAMNet output

    # Extract embeddings (assuming they are the second output)
    embeddings = yamnet_output[1].numpy()  # Typically, embeddings are in the second output
    return embeddings


def callback(indata, frames, time, status):
    global audio_buffer, predictions
    if status:
        print(status)

    # Convert the input data to float32 and normalize it
    audio_data = indata[:, 0]
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = audio_data

    if np.abs(audio_buffer).sum() > 0.1:  # Ensure there's enough signal to make a prediction
        # Extract embeddings and make a prediction
        embeddings = extract_yamnet_embeddings(audio_buffer)

        # Reshape embeddings to have the correct dimensions for the classifier model
        embeddings = np.expand_dims(embeddings, axis=0)  # Add batch dimension
        embeddings = embeddings[:, :2, :]  # Adjust the dimensions to match (None, 2, 1024)

        # Pass the reshaped embeddings to the classifier
        predicted_class = np.argmax(classifier_model.predict(embeddings))
        predictions.append(predicted_class)
        print(f"Prediction: {class_names[predicted_class]}")


def predict_real_time(duration_seconds=60):
    """Records audio for 'duration_seconds' and makes predictions every 3 seconds."""
    global predictions
    buffer_size = BUFFER_SIZE

    # Open an audio stream to record the input
    with sd.InputStream(channels=1, callback=callback, samplerate=SAMPLE_RATE, blocksize=BUFFER_SIZE):
        print("Recording and predicting in real-time...")
        sd.sleep(duration_seconds * 1000)  # Record for 'duration_seconds' milliseconds

    # Process all predictions (e.g., find the most common prediction)
    if predictions:
        unique, counts = np.unique(predictions, return_counts=True)
        most_common_prediction_index = unique[np.argmax(counts)]
        most_common_prediction_label = class_names[most_common_prediction_index]

        print("\nSummary of predictions:")
        for label_index, count in zip(unique, counts):
            print(f"Label {class_names[label_index]}: {count} occurrence(s)")

        print(f"\nMost common prediction over {duration_seconds // 60} minute(s): {most_common_prediction_label}")
    else:
        print("No predictions were made.")


# Start real-time prediction
predict_real_time(duration_seconds=30)  # Set to half a minute
