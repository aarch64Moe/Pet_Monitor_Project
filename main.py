import threading
import subprocess
import numpy as np
import time
import os
import cv2
import json
#Flask server
from flask import Flask, jsonify, Response, stream_with_context, request
from flask_cors import CORS
from werkzeug.serving import make_server
from threading import Event
import logging
#Sensors
import gps
import shared_data
from filters import bandpass_filter
from mpu import get_mpu6050_data, detect_movement
from calibration import get_calibration_data
from Temperuture_sensor import read_temperature_humidity
from heart import initialize_sensor, read_max30102, calculate_heart_rate
from DeviceNum import get_device_index
#Recoring and Yamnet
import sounddevice as sd
import tflite_runtime.interpreter as tflite
from sklearn.preprocessing import LabelEncoder
from collections import deque, Counter   # for most common label (Buffers)

#Global Definitions
terminate_flag = False  # Flag for stopping threads
recording = False
Gps_navigating = False
gps_thread = None  # Thread for GPS tracking

desired_device_name = "test"
device_index = get_device_index(desired_device_name)

# Paths and parameters
OUTPUT_DIR = "/home/radxa/yamnet/Recorded_audio_Debug"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sample_rate = 16000
segment_duration = 0.96  # .96-mill-second segments for processing
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
predictions_buffer = deque(maxlen=30)  # Stores the last 3 seconds of predictions (excluding "Noise")
segment_counter = 0

# Timer for limiting label display
last_display_time = 0  # Track last display time in seconds
display_interval = 2  # Display the most common label every 4 seconds

######################## Mobile App Flutter ##################################
recording_event = Event()
gps_event = Event()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Suppress Flask logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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
    global  recording
    recording = True  # Indicate recording started

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
            recording = False  # Ensure recording is reset
            print("Camera stream ended.")
            #enable_pins()  # Turn peripherals back on

    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/gps_coordinates', methods=['GET'])
def gps_coordinates():
    """Endpoint to retrieve the latest GPS coordinates."""
    global Gps_navigating

    if not Gps_navigating:
        Gps_navigating = True
        if not gps.gps_running:  # Ensure tracking isn't already running
            gps_thread_instance = threading.Thread(
                target=gps.start_gps_tracking,
                kwargs={"port": "/dev/ttyS2", "baudrate": 9600},
                daemon=True
            )
            gps_thread_instance.start()

    # Fetch the latest GPS coordinates
    coordinates = gps.get_current_coordinates()
    return jsonify(coordinates)

@app.route('/set_gps_navigating', methods=['POST'])
def set_gps_navigating():
    """Endpoint to control the GPS navigating state."""
    global Gps_navigating
    data = request.get_json()
    if 'Gps_navigating' in data:
        Gps_navigating = data['Gps_navigating'] # return from app true or false
        if not Gps_navigating:
            gps.stop_gps_tracking()  # Stop GPS tracking if navigating is set to False
        return jsonify({"status": "success", "Gps_navigating": Gps_navigating})
    return jsonify({"status": "error", "message": "Invalid request"}), 400

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

@app.route('/save_report', methods=['POST'])
def save_report():
    """Endpoint to save the monitor report."""
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data provided"}), 400

    # Log received data
    #print(f"Received report: {data}")

    try:
        file_path = "/home/radxa/examples/monitor_report.log"
        with open(file_path, "a") as log_file:
            log_file.write(f"{time.ctime()} - {json.dumps(data)}\n")
        return jsonify({"status": "success", "message": "Report saved"})
    except Exception as e:
        print(f"Error saving report: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reboot', methods=['POST'])
def reboot_device():
    os.system("sudo reboot")
    return jsonify({"status": "Rebooting..."})

class ServerThread(threading.Thread):
    """Runs Flask server in a separate thread."""
    def __init__(self, app):
        threading.Thread.__init__(self)
        self.server = make_server("0.0.0.0", 5000, app)
        self.context = app.app_context()
        self.context.push()

    def run(self):
        print("Starting Flask server...")
        self.server.serve_forever()

    def shutdown(self):
        print("Shutting down Flask server...")
        self.server.shutdown()

################################ Sensors and classification Loop ############################################

def main_sensor_loop(accel_offsets, gyro_offsets, prev_data):
    """Unified loop to handle temperature, MPU6050, and heart rate data."""
    global recording, Gps_navigating
    temperature_timer = time.time()
    mpu_timer = time.time()
    heart_rate_timer = time.time()
    classification_timer = time.time()  # Timer for real-time classification
    initialize_sensor()
    # Initial temperature and humidity reading
    try:
        temp, hum = read_temperature_humidity()
        # Read data from the heart rate sensor
        if temp is not None and hum is not None:
            shared_data.current_temperature = temp
            shared_data.current_humidity = hum
            print(f"Initial Temperature: {temp:.2f} °C, Initial Humidity: {hum:.2f} %")
        else:
            print("Failed to read initial temperature and humidity.")
    except Exception as e:
        print(f"Error in initial temperature reading: {e}")

    while not terminate_flag:
        if not recording and not Gps_navigating:
            # Real-time classification
            if time.time() - classification_timer >= 1:  # Every 1 second
                run_real_time_classification()
                classification_timer = time.time()

            # Update MPU6050 data
            if time.time() - mpu_timer >= 0.25:  # Every 0.25 seconds
                #print("Reading Data.")
                mpu_data = get_mpu6050_data(accel_offsets, gyro_offsets, prev_data)
                prev_data = mpu_data
                shared_data.movement = detect_movement()

                if shared_data.movement == "Moving":
                     shared_data.movement = "Moving"
                 #   print(f"Movement: {shared_data.movement}")

                mpu_timer = time.time()

            # Update temperature data
            if time.time() - temperature_timer >= 30:  # Every 30 seconds
                try:
                    temp, hum = read_temperature_humidity()
                    if temp is not None and hum is not None:
                        shared_data.current_temperature = temp
                        shared_data.current_humidity = hum
                        #print(f"Temperature: {temp:.2f} °C, Humidity: {hum:.2f} %")
                    else:
                        print("Failed to read temperature and humidity.")

                except Exception as e:
                    print(f"Error in temperature reading: {e}")

                temperature_timer = time.time()

            # Monitor heart rate when still
            if time.time() - heart_rate_timer >= 30:  # Every 30 seconds
                if shared_data.movement == "Standing Still" and shared_data.sitting_count >= 5:
                    try:
                        # Read data from the heart rate sensor
                        red_data, ir_data = read_max30102(duration=10)
                        if len(red_data) > 0 and len(ir_data) > 0:
                            # Calculate heart rate
                            heart_rate = calculate_heart_rate(red_data)
                            # Validate the heart rate value
                            if isinstance(heart_rate, (int, float)) and 40 <= heart_rate <= 220:
                                # Update shared data with a valid heart rate
                                shared_data.current_heart_rate = round(heart_rate, 2)
                                #print(f"Heart Rate: {shared_data.current_heart_rate:.2f} BPM")
                            else:
                                print(f"Invalid heart rate value: {heart_rate}. Resetting heart rate.")
                                shared_data.current_heart_rate = None  # Reset to None for invalid values
                        else:
                            print("No data from heart rate sensor.")
                    except Exception as e:
                        print(f"Error measuring heart rate: {e}")

                # Update the heart rate timer
                heart_rate_timer = time.time()

        else:
            # Wait during recording and Navigating GPS
            #print("Recording in progress, waiting...")
            recording_event.wait(timeout=3)  # Wait until recording ends or timeout
            gps_event.wait(timeout=3)

            if recording:  # Timeout occurred
                recording = False

            if Gps_navigating:  # Timeout occurred
                Gps_navigating = False

        time.sleep(0.1)  # Small sleep to reduce CPU usage

################################ Yamnet Model and recording ############################################

# Function to display the most common label every 4 seconds
def display_common_label():
    global last_display_time
    current_time = time.time()
    if current_time - last_display_time >= display_interval:
        # Get non-"Noise" labels from the buffer
        filtered_labels = [label for label in predictions_buffer if label != 'Noise']
        if filtered_labels:
            common_label = Counter(filtered_labels).most_common(1)[0][0]
            shared_data.current_classification = common_label
            #print(f"Most common label in the last 4 seconds: {common_label}")

        else:
            shared_data.current_classification = "Normal Behavior"

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

def run_real_time_classification():
    #print("Starting real-time classification...")
    global recording, device_index

    try:
        with sd.InputStream(device=device_index, callback=audio_callback, channels=1, samplerate=sample_rate,
                            blocksize=int(segment_duration * sample_rate)):
            #print("Processing audio segment...")
            time.sleep(0.1)  # Allow the input stream to process
    except Exception as e:
        print(f"Error in real-time classification: {e}")

    #print("Real-time classification finished.")

################################ Main Loop Operations ##########################################

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
    # Start Flask server in a separate thread
    server_thread = ServerThread(app)
    server_thread.start()

    # Start input listener in a thread
    threading.Thread(target=input_listener, daemon=True).start()

    # Wait for termination signal
    while not terminate_flag:
        time.sleep(0.1)

    # Stop Flask server
    server_thread.shutdown()

    # Exit the program
    print("Application terminated.")
    os._exit(0)
