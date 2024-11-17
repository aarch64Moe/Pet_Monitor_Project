from pickle import FALSE

import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import scipy.signal as signal
import threading
import time
from collections import deque, Counter
from sklearn.preprocessing import LabelEncoder
from scipy.io.wavfile import write
from flask import Flask, jsonify, Response, stream_with_context
import os
import cv2


from filters import apply_filter
from mpu import get_mpu6050_data, detect_movement
from calibration import get_calibration_data
import shared_data
from Temperuture_sensor import read_temperature_humidity
from heart import monitor_heart_rate_when_still




app = Flask(__name__)

# API endpoints
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        'temperature': shared_data.current_temperature,
        'humidity': shared_data.current_humidity,
        'heart_rate': shared_data.current_heart_rate,
        'movement': shared_data.current_movement,
        'classification': shared_data.current_classification,
        'is_running': shared_data.is_running
    })

# New reboot endpoint
@app.route('/reboot', methods=['POST'])
def reboot_device():
    os.system("sudo reboot")
    return jsonify({"status": "Rebooting..."})
import cv2
from flask import Response, stream_with_context




# Start the Flask server in a separate thread
def start_flask():
    app.run(host="0.0.0.0", port=5000)


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
predictions_buffer = deque(maxlen=30)  # Stores the last 4 seconds of predictions (excluding "Noise")
segment_counter = 0

# Flag to control the recording loop
recording = True

# Timer for limiting label display
last_display_time = 0  # Track last display time in seconds
display_interval = 4  # Display the most common label every 4 seconds


# Bandpass filter
def bandpass_filter(audio, sample_rate, lowcut, highcut):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.lfilter(b, a, audio)


# Function to display the most common label every 4 seconds
def display_common_label():
    global last_display_time
    current_time = time.time()
    if current_time - last_display_time >= display_interval:
        # Get non-"Noise" labels from the buffer
        filtered_labels = [label for label in predictions_buffer if label != 'Noise']
        if filtered_labels:
            common_label = Counter(filtered_labels).most_common(1)[0][0]
            print(f"Most common label in the last 4 seconds: {common_label}")
        # Reset the buffer every 4 seconds to avoid stale predictions
        predictions_buffer.clear()
        last_display_time = current_time  # Update the last display time


# Audio processing and prediction
def audio_callback(indata, frames, time, status):
    global segment_counter
    if status:
        print("Status Error:", status)

    audio_data = indata[:, 0]
    if np.all(audio_data == 0):
        print("Warning: Captured audio data is silent or empty.")
        return

    filtered_audio = bandpass_filter(audio_data, sample_rate, bandpass_lowcut, bandpass_highcut)

    if len(filtered_audio) >= int(segment_duration * sample_rate):
        segment = filtered_audio[:int(segment_duration * sample_rate)].astype(np.float32)
        segment_counter += 1

        with prediction_lock:
            shared_data.is_running = True
            yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], segment)
            yamnet_interpreter.invoke()
            embeddings = yamnet_interpreter.get_tensor(yamnet_output_details[1]['index']).mean(axis=0).astype(
                np.float32)
            classifier_interpreter.set_tensor(classifier_input_details[0]['index'], [embeddings])
            classifier_interpreter.invoke()
            prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

            # Only add non-"Noise" labels to predictions_buffer
            if predicted_label != 'Noise':
                predictions_buffer.append(predicted_label)
                shared_data.current_classification = predicted_label
            #print(f"Predicted label: {predicted_label}")
            display_common_label()  # Display the most common label every 4 seconds


# Start real-time classification with saved audio segments
def run_real_time_classification():
    print("Starting real-time classification...")
    device_index = 0  # Replace with the correct device index
    global recording
    with sd.InputStream(device=device_index, callback=audio_callback, channels=1, samplerate=sample_rate,
                        blocksize=int(segment_duration * sample_rate)):
        while recording:
            time.sleep(0.1)  # Small sleep to prevent CPU overload


    print("Real-time classification finished.")

# Function to display temperature every 10 seconds
def temperature_thread():
    while True:
        temp, hum = read_temperature_humidity()
        if temp is not None and hum is not None:
            shared_data.current_temperature = temp
            shared_data.current_humidity = hum
            print(f"Temperature: {temp:.2f} °C, Humidity: {hum:.2f} %")
        else:
            print("Failed to read temperature and humidity.")
        time.sleep(30)


# Main thread function to handle MPU6050 data
def mpu6050_thread(accel_offsets, gyro_offsets, prev_data):
    global mpu_data
    while True:
        mpu_data = get_mpu6050_data(accel_offsets, gyro_offsets, prev_data)
        prev_data = mpu_data
        shared_data.movement = detect_movement()

        if shared_data.movement == "Walking":
            print(f"Movement: {shared_data.movement}")

        time.sleep(0.5)

# Thread to handle user input for stopping
def monitor_input():
    global recording
    input("Press Enter to stop recording...\n")
    shared_data.is_running = False
    recording = False


# Run classification with input monitoring
if __name__ == "__main__":

    accel_offsets, gyro_offsets = get_calibration_data()

    # Start the mpu thread
    threading.Thread(target=mpu6050_thread, args=(accel_offsets, gyro_offsets, None), daemon=True).start()

    # Start the temperature thread
    threading.Thread(target=temperature_thread, daemon=True).start()

    # Start the heart rate monitoring thread
    threading.Thread(target=monitor_heart_rate_when_still, daemon=True).start()

    # Start the Flask server
    threading.Thread(target=start_flask, daemon=True).start()

    input_thread = threading.Thread(target=monitor_input)
    input_thread.start()
    run_real_time_classification()
    input_thread.join()
