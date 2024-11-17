import os
import time
import numpy as np
import subprocess
import threading
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import tflite_runtime.interpreter as tflite
from sklearn.preprocessing import LabelEncoder

# Parameters
RECORDING_DURATION = 5  # seconds
AMPLIFICATION_FACTOR = 40.0  # Amplify the audio
OUTPUT_DIR = "/home/radxa/yamnet/Recorded_audio_Test"
SAMPLE_RATE = 16000  # Lower sample rate
CHANNELS = 1  # Mono
stop_recording = False

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load TFLite models
yamnet_model_path = "/home/radxa/yamnet/model/Yamnet_Model.tflite"
classifier_model_path = "/home/radxa/yamnet/model/tflite_classifier_model.tflite"

yamnet_interpreter = tflite.Interpreter(model_path=yamnet_model_path)
yamnet_interpreter.allocate_tensors()
classifier_interpreter = tflite.Interpreter(model_path=classifier_model_path)
classifier_interpreter.allocate_tensors()

# Get model input and output details
yamnet_input_details = yamnet_interpreter.get_input_details()
yamnet_output_details = yamnet_interpreter.get_output_details()
classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

# Label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Angry', 'Fighting', 'Happy', 'HuntingMind', 'Noise', 'Purring'])

def record_audio(filename):
    command = [
        'arecord', '-D', 'plughw:0', '-c', str(CHANNELS),
        '-r', str(SAMPLE_RATE), '-f', 'S16_LE', '-t', 'wav',
        '-d', str(RECORDING_DURATION),
        '-V', 'mono', '-v', filename
    ]
    # Redirect arecord output to /dev/null
    with open(os.devnull, 'w') as f:
        result = subprocess.run(command, stdout=f, stderr=f)
    return result.returncode == 0

def amplify_audio(audio_data):
    return np.clip(audio_data * AMPLIFICATION_FACTOR, -32768, 32767).astype(np.int16)

def apply_bandpass_filter(audio_data, lowcut=300.0, highcut=7500.0, sample_rate=SAMPLE_RATE):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, audio_data)

def apply_noise_gate(audio_data, threshold=0.02):
    audio_data[np.abs(audio_data) < threshold] = 0
    return audio_data

def prepare_audio_for_yamnet(audio_data):
    # Ensure the audio is in float32 format and matches YAMNet’s expected shape (1, 15360)
    audio_data = audio_data.astype(np.float32)
    if SAMPLE_RATE != 16000:
        # Resample to 15360 samples (0.96 seconds at 16 kHz)
        audio_data = np.interp(np.linspace(0, len(audio_data), 15360), np.arange(len(audio_data)), audio_data)
    return np.reshape(audio_data[:15360], (1, 15360))

def extract_yamnet_embeddings(audio_data):
    # Reshape audio data to a single dimension (15360,)
    segment = np.reshape(audio_data, (15360,)).astype(np.float32)
    yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], segment)
    yamnet_interpreter.invoke()
    embeddings = yamnet_interpreter.get_tensor(yamnet_output_details[1]['index']).mean(axis=0).astype(np.float32)
    print("Extracted embeddings:", embeddings)  # Debugging print for embeddings
    return embeddings

def classify_audio(embeddings):
    classifier_interpreter.set_tensor(classifier_input_details[0]['index'], [embeddings])
    classifier_interpreter.invoke()
    prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])
    print("Prediction probabilities:", prediction)  # Debugging print for prediction probabilities
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

def process_and_save_audio(input_filename, output_filename):
    sample_rate, audio_data = wavfile.read(input_filename)
    audio_data = apply_noise_gate(audio_data.astype(np.float32))
    amplified_audio = amplify_audio(audio_data)
    filtered_audio = apply_bandpass_filter(amplified_audio)

    # Prepare and classify
    yamnet_input_audio = prepare_audio_for_yamnet(filtered_audio)
    embeddings = extract_yamnet_embeddings(yamnet_input_audio)
    prediction = classify_audio(embeddings)

    # Save processed audio and output prediction
    wavfile.write(output_filename, sample_rate, filtered_audio.astype(np.int16))
    print(f"Audio saved to {output_filename} with prediction: {prediction}")


def stop_recording_listener():
    global stop_recording
    input("Press Enter to stop recording...\n")
    stop_recording = True

def main():
    global stop_recording
    print("Recording... Press Enter to stop.")
    segment_count = 1

    listener_thread = threading.Thread(target=stop_recording_listener)
    listener_thread.start()

    try:
        while not stop_recording:
            raw_filename = os.path.join(OUTPUT_DIR, "temp_recording.wav")
            output_filename = os.path.join(OUTPUT_DIR, f"recording_{segment_count}_{int(time.time())}.wav")

            if not record_audio(raw_filename):
                print("Error: Unable to access recording device.")
                break

            process_and_save_audio(raw_filename, output_filename)
            segment_count += 1

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stop_recording = True
        listener_thread.join()
        print("Recording stopped.")

if __name__ == "__main__":
    main()
