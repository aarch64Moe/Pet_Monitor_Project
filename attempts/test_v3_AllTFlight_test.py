import numpy as np
import tflite_runtime.interpreter as tflite
import scipy.signal as signal
import time
from collections import deque, Counter
from sklearn.preprocessing import LabelEncoder
import subprocess
import os
from scipy.io import wavfile

# Paths and parameters
OUTPUT_DIR = "/home/radxa/yamnet/Recorded_audio_Test"
sample_rate = 16000
total_duration = 120  # 2 minutes
display_interval = 30
segment_duration = 0.96
bandpass_lowcut = 300
bandpass_highcut = 5000
AMPLIFICATION_FACTOR = 40.0
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the TFLite models
yamnet_interpreter = tflite.Interpreter(model_path="/home/radxa/yamnet/model/Yamnet_Model.tflite")
yamnet_interpreter.allocate_tensors()
classifier_interpreter = tflite.Interpreter(model_path="/home/radxa/yamnet/model/tflite_classifier_model.tflite")
classifier_interpreter.allocate_tensors()

# Get model input and output details
yamnet_input_details = yamnet_interpreter.get_input_details()
yamnet_output_details = yamnet_interpreter.get_output_details()
classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

# Label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Angry', 'Fighting', 'Happy', 'HuntingMind', 'Noise', 'Purring'])

predictions_buffer = deque(maxlen=int(display_interval / segment_duration))


# Band-pass filter function
def bandpass_filter(audio, sample_rate, lowcut, highcut):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.lfilter(b, a, audio)


# Amplify function
def amplify_audio(audio_data):
    return np.clip(audio_data * AMPLIFICATION_FACTOR, -32768, 32767).astype(np.int16)


def record_audio(filename):
    command = [
        'arecord', '-D', 'plughw:1', '-c', '1',
        '-r', str(sample_rate), '-f', 'S16_LE', '-t', 'wav',
        '-d', str(int(segment_duration)), '-V', 'mono', '-v', filename
    ]
    with open(os.devnull, 'w') as f:
        result = subprocess.run(command, stdout=f, stderr=f)

    if result.returncode == 0:
        print(f"Recording saved: {filename}")
    else:
        print(f"Recording failed for {filename}")

    return result.returncode == 0

def process_and_classify(filename):
    # Read and process the audio data
    sample_rate, audio_data = wavfile.read(filename)
    filtered_audio = bandpass_filter(audio_data, sample_rate, bandpass_lowcut, bandpass_highcut)
    amplified_audio = amplify_audio(filtered_audio)

    # Resample and prepare for YAMNet
    yamnet_input_audio = np.interp(
        np.linspace(0, len(amplified_audio), 15360),
        np.arange(len(amplified_audio)),
        amplified_audio.astype(np.float32)
    ).reshape((15360,)).astype(np.float32)

    # YAMNet prediction
    yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], yamnet_input_audio)
    yamnet_interpreter.invoke()
    embeddings = yamnet_interpreter.get_tensor(yamnet_output_details[1]['index']).mean(axis=0).astype(np.float32)

    # Classifier prediction
    classifier_interpreter.set_tensor(classifier_input_details[0]['index'], [embeddings])
    classifier_interpreter.invoke()
    prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    predictions_buffer.append(predicted_label)
    print(f"Prediction: {predicted_label}")


def display_most_common_class():
    if predictions_buffer:
        common_class = Counter(predictions_buffer).most_common(1)[0][0]
        print(f"Most common class in the last {display_interval} seconds: {common_class}")
    else:
        print("No predictions in the last 30 seconds.")


# Main loop for classification
def run_classification_loop():
    start_time = time.time()
    next_display_time = start_time + display_interval
    segment_count = 1

    print("Starting classification...")
    while time.time() - start_time < total_duration:
        filename = os.path.join(OUTPUT_DIR, f"recorded_segment_{segment_count}_{int(time.time())}.wav")
        if record_audio(filename):
            process_and_classify(filename)
            segment_count += 1

        if time.time() >= next_display_time:
            display_most_common_class()
            next_display_time += display_interval

    print("Classification loop finished.")


if __name__ == "__main__":
    run_classification_loop()
