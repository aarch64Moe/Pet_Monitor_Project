import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import scipy.signal as signal
import subprocess
import time
from collections import Counter, deque
from sklearn.preprocessing import LabelEncoder
from scipy.io import wavfile
import os

# Paths and parameters
OUTPUT_DIR = "/home/radxa/yamnet/Recorded_audio_Debug"
sample_rate = 16000
total_duration = 2 * 60  # 2 minutes
segment_duration = 1  # Process in 1-second chunks
record_duration = 5  # Record in 5-second chunks
bandpass_lowcut = 300
bandpass_highcut = 5000
AMPLIFICATION_FACTOR = 80.0  # Increased for better audio level
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the TFLite models
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

predictions_buffer = deque(maxlen=int(record_duration / segment_duration))

def amplify_audio(audio_data):
    return np.clip(audio_data * AMPLIFICATION_FACTOR, -32768, 32767).astype(np.int16)

def bandpass_filter(audio, sample_rate, lowcut, highcut):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.lfilter(b, a, audio)

def record_audio(filename, duration):
    command = [
        'arecord', '-D', 'plughw:0', '-c', '1',
        '-r', str(sample_rate), '-f', 'S16_LE', '-t', 'wav',
        '-d', str(duration), filename
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Recording saved: {filename}")
    else:
        print(f"Recording failed: {result.stderr}")
    return result.returncode == 0

def process_and_classify_segment(segment):
    #segment = amplify_audio(segment)
    #segment = bandpass_filter(segment, sample_rate, bandpass_lowcut, bandpass_highcut)
    yamnet_input = np.interp(
        np.linspace(0, len(segment), 15360),
        np.arange(len(segment)),
        segment.astype(np.float32)
    ).reshape((15360,)).astype(np.float32)

    yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], yamnet_input)
    yamnet_interpreter.invoke()
    embeddings = yamnet_interpreter.get_tensor(yamnet_output_details[1]['index']).mean(axis=0).astype(np.float32)

    classifier_interpreter.set_tensor(classifier_input_details[0]['index'], [embeddings])
    classifier_interpreter.invoke()
    prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    predictions_buffer.append(predicted_label)
    print(f"Predicted label: {predicted_label}")

def run_classification_loop():
    start_time = time.time()
    segment_count = 1

    while time.time() - start_time < total_duration:
        filename = os.path.join(OUTPUT_DIR, f"recorded_segment_{segment_count}_{int(time.time())}.wav")
        if record_audio(filename, record_duration):
            sample_rate, audio_data = wavfile.read(filename)
            for i in range(0, len(audio_data), sample_rate):  # Process 1-second chunks
                segment = audio_data[i:i + sample_rate]
                if len(segment) == sample_rate:
                    process_and_classify_segment(segment)
            segment_count += 1
    print("Classification loop finished.")

if __name__ == "__main__":
    run_classification_loop()
