from pickle import FALSE


import tflite_runtime.interpreter as tflite
import scipy.signal as signal


from filters import apply_filter
from mpu import get_mpu6050_data, detect_movement
from calibration import get_calibration_data
import shared_data
from Temperuture_sensor import read_temperature_humidity
from heart import monitor_heart_rate_when_still

from flask import Flask, jsonify, Response, stream_with_context
import sounddevice as sd
import threading
import time
import numpy as np
from collections import deque, Counter
from sklearn.preprocessing import LabelEncoder
from scipy.signal import butter, lfilter
from scipy.io.wavfile import write
import cv2
import os
from mpu import get_mpu6050_data, detect_movement
from calibration import get_calibration_data
from Temperuture_sensor import read_temperature_humidity
from heart import monitor_heart_rate_when_still
import shared_data

# Global flags
recording = True
paused = False
recording_active = False
lock = threading.Lock()

# Flask app setup
app = Flask(__name__)

# Audio processing configuration
OUTPUT_DIR = "/home/radxa/yamnet/Recorded_audio_Debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sample_rate = 16000
segment_duration = 0.96
bandpass_lowcut = 300
bandpass_highcut = 5000
prediction_lock = threading.Lock()

# TFLite model setup
yamnet_interpreter = tflite.Interpreter(model_path="/home/radxa/yamnet/model/Yamnet_Model.tflite")
yamnet_interpreter.allocate_tensors()
classifier_interpreter = tflite.Interpreter(model_path="/home/radxa/yamnet/model/tflite_classifier_model.tflite")
classifier_interpreter.allocate_tensors()

yamnet_input_details = yamnet_interpreter.get_input_details()
yamnet_output_details = yamnet_interpreter.get_output_details()
classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Angry', 'Fighting', 'Happy', 'HuntingMind', 'Noise', 'Purring'])
predictions_buffer = deque(maxlen=30)
last_display_time = 0
display_interval = 4

# Bandpass filter
def bandpass_filter(audio, sample_rate, lowcut, highcut):
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, audio)

# Audio callback
def audio_callback(indata, frames, time_info, status):
    global recording_active
    if status:
        print("Status Error:", status)

    if paused:
        return

    audio_data = indata[:, 0]
    filtered_audio = bandpass_filter(audio_data, sample_rate, bandpass_lowcut, bandpass_highcut)
    segment = filtered_audio[:int(segment_duration * sample_rate)].astype(np.float32)

    with prediction_lock:
        yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], segment)
        yamnet_interpreter.invoke()
        embeddings = yamnet_interpreter.get_tensor(yamnet_output_details[1]['index']).mean(axis=0)
        classifier_interpreter.set_tensor(classifier_input_details[0]['index'], [embeddings])
        classifier_interpreter.invoke()
        prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        if predicted_label != "Noise":
            predictions_buffer.append(predicted_label)

# Start/stop audio stream
def start_audio_stream():
    global recording_active
    if not recording_active:
        print("Starting audio stream...")
        sd.InputStream(device=0, callback=audio_callback, channels=1, samplerate=sample_rate).start()
        recording_active = True

def stop_audio_stream():
    global recording_active
    if recording_active:
        print("Stopping audio stream...")
        sd.stop()
        recording_active = False

# Flask endpoints
@app.route('/pause_classification', methods=['POST'])
def pause_classification():
    global paused
    with lock:
        paused = True
        stop_audio_stream()
    print("Classification paused.")
    return jsonify({"status": "Paused"})

@app.route('/resume_classification', methods=['POST'])
def resume_classification():
    global paused
    with lock:
        paused = False
        start_audio_stream()
    print("Classification resumed.")
    return jsonify({"status": "Resumed"})

@app.route('/camera_stream', methods=['GET'])
def camera_stream():
    global paused
    paused = True
    stop_audio_stream()

    def generate():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            cap.release()
            start_audio_stream()
            paused = False
            print("Camera stream ended.")

    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main loop
def main_loop():
    accel_offsets, gyro_offsets = get_calibration_data()
    temp_timer = time.time()
    mpu_timer = time.time()
    start_audio_stream()

    while recording:
        with lock:
            if not paused:
                if time.time() - mpu_timer > 0.5:
                    mpu_data = get_mpu6050_data(accel_offsets, gyro_offsets, None)
                    shared_data.movement = detect_movement()
                    mpu_timer = time.time()

                if time.time() - temp_timer > 30:
                    temp, hum = read_temperature_humidity()
                    if temp is not None and hum is not None:
                        shared_data.current_temperature = temp
                        shared_data.current_humidity = hum
                    temp_timer = time.time()
            else:
                time.sleep(1)

    stop_audio_stream()

# Input monitor
def monitor_input():
    global recording
    input("Press Enter to stop recording...\n")
    recording = False

# Start Flask server
def start_flask():
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    threading.Thread(target=start_flask, daemon=True).start()
    threading.Thread(target=monitor_heart_rate_when_still, daemon=True).start()
    monitor_input_thread = threading.Thread(target=monitor_input)
    monitor_input_thread.start()
    main_loop()
    monitor_input_thread.join()