import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import scipy.signal as signal
import time
from collections import deque, Counter
from sklearn.preprocessing import LabelEncoder

# Load the TFLite YAMNet model
yamnet_interpreter = tflite.Interpreter(model_path="/home/radxa/yamnet/model/Yamnet_Model.tflite")
yamnet_interpreter.allocate_tensors()

# Load the TFLite classifier model
classifier_interpreter = tflite.Interpreter(model_path="/home/radxa/yamnet/model/tflite_classifier_model.tflite")
classifier_interpreter.allocate_tensors()

# Get input and output details
yamnet_input_details = yamnet_interpreter.get_input_details()
yamnet_output_details = yamnet_interpreter.get_output_details()
classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

# Label encoder to map prediction indices to class names
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Angry', 'Fighting', 'Happy', 'HuntingMind', 'Noise', 'Purring'])

# Update this with your actual device index and number of channels
INPUT_DEVICE_INDEX = 0  # Replace with your input device index
INPUT_CHANNELS = 1     # Replace with the number of channels your device supports


# Parameters
sample_rate = 16000
total_duration = 2 * 60
display_interval = 30
segment_duration = 0.96
bandpass_lowcut = 300
bandpass_highcut = 5000

# Buffer to store predictions
predictions_buffer = deque(maxlen=int(display_interval / segment_duration))

# Band-pass filter function
def bandpass_filter(audio, sample_rate, lowcut, highcut):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_audio = signal.lfilter(b, a, audio)
    return filtered_audio

# Function to process real-time audio and make predictions
def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    # If your device provides multiple channels, select one or average them
    if INPUT_CHANNELS > 1:
        audio_data = indata.mean(axis=1)  # Averages all channels
    else:
        audio_data = indata[:, 0]

    #audio_data = indata[:, 0]
    filtered_audio = bandpass_filter(audio_data, sample_rate, bandpass_lowcut, bandpass_highcut)

    # Ensure we have enough data
    if len(filtered_audio) >= int(segment_duration * sample_rate):

        segment = filtered_audio[:int(segment_duration * sample_rate)].astype(np.float32)

        # Set the YAMNet model input with shape [15360]
        yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], segment)

        # Run YAMNet inference
        yamnet_interpreter.invoke()

        # Extract embeddings
        embeddings = yamnet_interpreter.get_tensor(yamnet_output_details[1]['index']).mean(axis=0).astype(np.float32)

        # Set embeddings as input to the classifier model
        classifier_interpreter.set_tensor(classifier_input_details[0]['index'], [embeddings])

        # Run classifier inference
        classifier_interpreter.invoke()

        # Get the predicted label
        prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])
        predicted_label_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

        # Add prediction to buffer
        predictions_buffer.append(predicted_label)

# Function to display the most common class every 30 seconds
def display_most_common_class():
    if len(predictions_buffer) > 0:
        common_class = Counter(predictions_buffer).most_common(1)[0][0]
        print(f"Most common class in the last 30 seconds: {common_class}")
    else:
        print("No predictions in the last 30 seconds.")

# Run real-time classification
def run_real_time_classification(duration=total_duration, display_interval=30):
    print("Starting real-time classification...")
    start_time = time.time()
    next_display_time = start_time + display_interval

    # Use device=0 for input based on your system's available devices
    with sd.InputStream(callback=audio_callback, channels=INPUT_CHANNELS, samplerate=sample_rate,
                        blocksize=int(segment_duration * sample_rate), device=INPUT_DEVICE_INDEX):
        while time.time() - start_time < duration:
            if time.time() >= next_display_time:
                display_most_common_class()
                next_display_time += display_interval

    print("Real-time classification finished.")

# Start the real-time classification
run_real_time_classification()
