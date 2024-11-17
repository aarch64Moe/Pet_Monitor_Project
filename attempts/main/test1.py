import tensorflow as tf
import numpy as np
import sounddevice as sd

# Load YAMNet model (feature extractor)
yamnet_model_path = '/home/moe/PycharmProjects/Download_preparev2/yamnet_model/'
yamnet_model = tf.saved_model.load(yamnet_model_path)

# Load fine-tuned classifier model
classifier_model_path = '/home/moe/PycharmProjects/Download_preparev2/fine_tuned_classifier'
classifier_model = tf.keras.models.load_model(classifier_model_path)

# Sampling rate and duration
sample_rate = 16000  # YAMNet expects 16kHz audio
duration = 3  # 3 seconds for prediction windows
buffer_size = sample_rate * duration  # Total samples for 3 seconds


def extract_yamnet_embeddings(audio_buffer):
    # Convert the buffer into a tensor and flatten it
    input_tensor = tf.convert_to_tensor(audio_buffer, dtype=tf.float32)
    input_tensor = tf.reshape(input_tensor, [buffer_size])  # Flatten the buffer to shape (48000,)

    # Run the YAMNet model to extract embeddings
    scores, embeddings, spectrogram = yamnet_model(input_tensor)

    # Average the embeddings across time frames
    averaged_embeddings = tf.reduce_mean(embeddings, axis=0)

    # Reshape to match the expected input shape of the classifier (None, 2, 1024)
    reshaped_embeddings = tf.tile(tf.expand_dims(averaged_embeddings, axis=0), [2, 1])

    return reshaped_embeddings


# Function to predict in real-time
def predict_real_time(duration_minutes=1):
    # Initialize buffer to hold the audio data
    audio_buffer = np.zeros((buffer_size,), dtype=np.float32)

    print("Recording and predicting in real-time...")

    predictions = []

    # Callback to process each audio chunk in real-time
    def callback(indata, frames, time, status):
        nonlocal audio_buffer
        audio_chunk = indata[:, 0]  # Take mono channel data
        audio_buffer = np.roll(audio_buffer, -len(audio_chunk))  # Shift buffer left
        audio_buffer[-len(audio_chunk):] = audio_chunk  # Insert new audio chunk at the end

        if len(audio_buffer) == buffer_size:
            # Step 1: Extract YAMNet embeddings from the audio buffer
            embeddings = extract_yamnet_embeddings(audio_buffer)

            # Step 2: Use the fine-tuned classifier model to make a prediction
            scores = classifier_model.predict(np.array([embeddings]))

            # Get the label with the highest score
            predicted_label = np.argmax(scores, axis=-1)[0]
            predictions.append(predicted_label)

            # Print the predicted label (you can map this to actual class names if needed)
            print(f"Prediction: {predicted_label}")

    # Start the audio stream and process predictions for the specified duration
    with sd.InputStream(channels=1, samplerate=sample_rate, callback=callback):
        sd.sleep(int(duration_minutes * 60 * 1000))  # Run for 'duration_minutes' minutes

    # Process all predictions (e.g., find the most common prediction)
    if predictions:
        unique, counts = np.unique(predictions, return_counts=True)
        most_common_prediction = unique[np.argmax(counts)]
        print(f"Most common prediction over {duration_minutes} minute(s): {most_common_prediction}")
    else:
        print("No predictions were made.")


# Run the prediction for 1 minute
predict_real_time(duration_minutes=1)
