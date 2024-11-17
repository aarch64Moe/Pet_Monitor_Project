from flask import Flask, jsonify, Response, stream_with_context
from flask_cors import CORS
import cv2
import threading
import time
import os
import shared_data
from filters import apply_filter
from mpu import get_mpu6050_data, detect_movement
from calibration import get_calibration_data
from Temperuture_sensor import read_temperature_humidity
from heart import monitor_heart_rate_when_still
import sounddevice as sd
import numpy as np
import tflite_runtime.interpreter as tflite
from scipy.io.wavfile import write
from sklearn.preprocessing import LabelEncoder
from collections import deque, Counter
import scipy.signal as signal

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

terminate_flag = False  # Flag for stopping threads
paused = False
recording = False

# Flask Endpoints
@app.route('/')
def index():
    # Simple homepage with video stream
    return '''
    <html>
        <head>
            <title>Radxa Camera Stream</title>
        </head>
        <body>
            <h1>Camera Stream</h1>
            <img src="/camera_stream" style="width: 100%; height: auto;">
        </body>
    </html>
    '''

@app.route('/camera_stream', methods=['GET'])
def camera_stream():
    global paused
    paused = True
    global recording
    recording = True

    def generate():
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Error: Could not open the camera.")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)

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
            recording = False
            paused = False
            print("Camera stream ended.")

    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame')

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

@app.route('/pause_classification', methods=['POST'])
def pause_classification():
    global recording
    recording = True
    return jsonify({"status": "Classification paused"})

@app.route('/reboot', methods=['POST'])
def reboot_device():
    os.system("sudo reboot")
    return jsonify({"status": "Rebooting..."})

# Thread Functions
def main_sensor_loop(accel_offsets, gyro_offsets, prev_data):
    """Unified loop to handle temperature and MPU6050 data."""
    global recording
    temperature_timer = time.time()
    mpu_timer = time.time()

    while not terminate_flag:  # Keep running unless terminated
        if not recording:
            # Update MPU6050 data
            if time.time() - mpu_timer >= 0.5:
                mpu_data = get_mpu6050_data(accel_offsets, gyro_offsets, prev_data)
                prev_data = mpu_data
                shared_data.movement = detect_movement()

                if shared_data.movement == "Walking":
                    print(f"Movement: {shared_data.movement}")

                mpu_timer = time.time()

            # Update temperature data
            if time.time() - temperature_timer >= 30:
                try:
                    #monitor_heart_rate_when_still()
                    temp, hum = read_temperature_humidity()
                    if temp is not None and hum is not None:
                        shared_data.current_temperature = temp
                        shared_data.current_humidity = hum
                        print(f"Temperature: {temp:.2f} °C, Humidity: {hum:.2f} %")
                    else:
                        print("Failed to read temperature and humidity.")
                except Exception as e:
                    print(f"Error in temperature reading: {e}")

                temperature_timer = time.time()
        else:
            # Log when waiting for recording to finish
            print("Recording in progress, waiting...")
            time.sleep(0.1)  # Small sleep to reduce CPU usage


def input_listener():
    """Listens for termination signal."""
    global terminate_flag
    input("Press Enter to terminate the application...\n")
    terminate_flag = True
    print("Termination signal received.")

# Main Execution
if __name__ == "__main__":
    accel_offsets, gyro_offsets = get_calibration_data()

    # Start threads
    threading.Thread(target=main_sensor_loop, args=(accel_offsets, gyro_offsets, None), daemon=True).start()
    threading.Thread(target=input_listener, daemon=True).start()

    # Start Flask server
    app.run(host="0.0.0.0", port=5000)
