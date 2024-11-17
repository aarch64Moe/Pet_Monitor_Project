import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import scipy.signal as signal
import threading
import time
from collections import deque, Counter
from sklearn.preprocessing import LabelEncoder
from scipy.io.wavfile import write
import os

# Paths and parameters
OUTPUT_DIR = "/home/radxa/yamnet/Recorded_audio_Debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sample_rate = 16000
segment_duration = 0.96  # 1-second segments for processing
bandpass_lowcut = 300
bandpass_highcut = 5000
prediction_lock = threading.Lock()

# Load TFLite models
yamnet_interpreter = tflite.Interpreter(model_path="/home/radxa/yamnet/model/Yamnet_Model.tflite")
yamnet_interpreter.allocate_tensors()
classifier_interpreter = tflite.Interpreter(model_path="/home/radxa/yamnet/model/tflite_classifier_model.tflite")
classifier_interpreter.allocate_tensors()

# Model input/output details
yamnet_input_details = yamnet_interpreter.get_input_details()
yamnet_output_details = yamnet_interpreter.get_output_details()
classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

# Label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Angry', 'Fighting', 'Happy', 'HuntingMind', 'Noise', 'Purring'])

# Buffer for predictions
predictions_buffer = deque(maxlen=30)
segment_counter = 0


# Bandpass filter
def bandpass_filter(audio, sample_rate, lowcut, highcut):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.lfilter(b, a, audio)


# Audio processing and prediction
def audio_callback(indata, frames, time, status):
    global segment_counter
    if status:
        print("Status Error:", status)

    audio_data = indata[:, 0]

    # Print audio data to confirm it contains values
    #print("Captured audio data:", audio_data[:10])  # Show first 10 samples for debugging

    # Check if audio_data is empty or all zeros
    if np.all(audio_data == 0):
        print("Warning: Captured audio data is silent or empty.")
        return

    filtered_audio = bandpass_filter(audio_data, sample_rate, bandpass_lowcut, bandpass_highcut)

    if len(filtered_audio) >= int(segment_duration * sample_rate):
        segment = filtered_audio[:int(segment_duration * sample_rate)].astype(np.float32)

        # Save audio segment for debugging
        segment_filename = os.path.join(OUTPUT_DIR, f"audio_segment_{segment_counter}.wav")
        write(segment_filename, sample_rate, segment.astype(np.int16))
        print(f"Saved audio segment: {segment_filename}")
        segment_counter += 1

        with prediction_lock:
            # Ensure model processes each segment in order
            yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], segment)
            yamnet_interpreter.invoke()

            # Extract embeddings from YAMNet
            embeddings = yamnet_interpreter.get_tensor(yamnet_output_details[1]['index']).mean(axis=0).astype(
                np.float32)

            # Classify with embeddings
            classifier_interpreter.set_tensor(classifier_input_details[0]['index'], [embeddings])
            classifier_interpreter.invoke()

            # Get predicted label
            prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            predictions_buffer.append(predicted_label)
            print(f"Predicted label: {predicted_label}")


# Start real-time classification with saved audio segments
def run_real_time_classification():
    print("Starting real-time classification...")
    device_index = 0 # Replace with the correct device index
    with sd.InputStream(device=device_index, callback=audio_callback, channels=1, samplerate=sample_rate,
                        blocksize=int(segment_duration * sample_rate)):
        time.sleep(120)  # Run for 2 minutes
    print("Real-time classification finished.")


# Run classification
run_real_time_classification()
