import smbus2
import time
import numpy as np
from datetime import datetime
import os
import sounddevice as sd
import soundfile as sf
import tflite_runtime.interpreter as tflite
import joblib
from sklearn.preprocessing import LabelEncoder
from scipy.signal import resample
import threading
from filters import apply_filter
from mpu import get_mpu6050_data, detect_movement
from calibration import get_calibration_data
import shared_data
from Temperuture_sensor import read_temperature_humidity
from heart import monitor_heart_rate_when_still

# Paths
yamnet_model_path = "/home/radxa/yamnet/model/Yamnet_Model.tflite"
classifier_model_path = "/home/radxa/yamnet/model/tflite_classifier_model.tflite"
output_dir = f"/home/radxa/yamnet/classified_audio_{datetime.now().strftime('%Y%m%d')}"
os.makedirs(output_dir, exist_ok=True)

# Load YAMNet and classifier models using TFLite runtime
yamnet_interpreter = tflite.Interpreter(model_path=yamnet_model_path)
yamnet_interpreter.allocate_tensors()

classifier_interpreter = tflite.Interpreter(model_path=classifier_model_path)
classifier_interpreter.allocate_tensors()

# Get input and output details for YAMNet and classifier
yamnet_input_details = yamnet_interpreter.get_input_details()
yamnet_output_details = yamnet_interpreter.get_output_details()
classifier_input_details = classifier_interpreter.get_input_details()
classifier_output_details = classifier_interpreter.get_output_details()

# Define label encoder and class names
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Angry', 'Fighting', 'Happy', 'HuntingMind', 'Noise', 'Purring'])

# Audio parameters
SAMPLE_RATE = 48000
TARGET_SAMPLE_RATE = 16000
RECORDING_DURATION = 1
SEGMENT_DURATION = 0.96  # Segment length for YAMNet

# Amplify the audio
def amplify_audio(data, db_gain=30):
    factor = 10 ** (db_gain / 20)
    return data * factor
# Resample audio to 16 kHz
def resample_audio(audio_data, orig_samplerate, target_samplerate=16000):
    number_of_samples = int(len(audio_data) * float(target_samplerate) / orig_samplerate)
    return resample(audio_data, number_of_samples)

# Prepare audio for YAMNet with 0.96-second segments
def prepare_audio_for_yamnet(audio_data, orig_samplerate):
    if len(audio_data.shape) > 1:  # Convert to mono if stereo
        audio_data = np.mean(audio_data, axis=1)
    resampled_audio = resample_audio(audio_data, orig_samplerate, TARGET_SAMPLE_RATE)
    # Ensure length matches YAMNet's expected input (15360 samples)
    segment_length = yamnet_input_details[0]['shape'][0]
    return resampled_audio[:segment_length].astype(np.float32)

# Extract YAMNet embeddings
def extract_yamnet_embeddings(audio_data):
    segment = np.reshape(audio_data, yamnet_input_details[0]['shape']).astype(np.float32)
    yamnet_interpreter.set_tensor(yamnet_input_details[0]['index'], segment)
    yamnet_interpreter.invoke()
    embeddings = yamnet_interpreter.get_tensor(yamnet_output_details[1]['index']).mean(axis=0).astype(np.float32)
    return embeddings

# Function to record audio and classify
def record_audio_and_process():
    with shared_data.lock:
        print(f"Recording audio for {RECORDING_DURATION} second(s)...")

        # Record audio
        audio_data = sd.rec(int(RECORDING_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        # Amplify and filter audio
        audio_data = amplify_audio(audio_data, 40)
        filtered_audio = apply_filter(audio_data.flatten(), 'butterworth', fs=SAMPLE_RATE)

        # Prepare for YAMNet and extract embeddings
        yamnet_input_audio = prepare_audio_for_yamnet(filtered_audio, SAMPLE_RATE)
        embeddings = extract_yamnet_embeddings(yamnet_input_audio)

        # Predict using the TFLite classifier
        classifier_interpreter.set_tensor(classifier_input_details[0]['index'], [embeddings])
        classifier_interpreter.invoke()
        prediction = classifier_interpreter.get_tensor(classifier_output_details[0]['index'])
        predicted_label_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]

        # Print prediction and save audio with timestamp
        print(f"Prediction: {predicted_label}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_audio_filename = os.path.join(output_dir, f"processed_voice_{timestamp}_{predicted_label}.wav")
        sf.write(processed_audio_filename, filtered_audio, SAMPLE_RATE)

        print(f"Processed audio saved: {processed_audio_filename}")

        # Reset flag after processing
        shared_data.recording_flag = False
        shared_data.last_recording_time = time.time()

# Function to display temperature every 10 seconds
def temperature_thread():
    while True:
        temp, hum = read_temperature_humidity()
        if temp is not None and hum is not None:
            print(f"Temperature: {temp:.2f} °C, Humidity: {hum:.2f} %")
        else:
            print("Failed to read temperature and humidity.")
        time.sleep(10)


# Main thread function to handle MPU6050 data
def mpu6050_thread(accel_offsets, gyro_offsets, prev_data):
    global mpu_data
    while True:
        mpu_data = get_mpu6050_data(accel_offsets, gyro_offsets, prev_data)
        prev_data = mpu_data
        shared_data.movement = detect_movement()

        if shared_data.movement == "Walking":
            print(f"Movement: {shared_data.movement}")

        time.sleep(0.1)


# Main function
def main():
    print("Loading calibration data...")
    accel_offsets, gyro_offsets = get_calibration_data()
    print("Calibration data loaded.")

    # Start the MPU6050 thread for movement detection
    print("Starting MPU6050 thread...")
    threading.Thread(target=mpu6050_thread, args=(accel_offsets, gyro_offsets, None), daemon=True).start()

    # Start the temperature monitoring thread
    print("Starting temperature monitoring thread...")
    threading.Thread(target=temperature_thread, daemon=True).start()

    # Start the heart rate monitoring thread
    print("Starting heart rate monitoring thread...")
    threading.Thread(target=monitor_heart_rate_when_still, daemon=True).start()

    # Start the audio recording and classification thread
    print("Starting audio recording and classification thread...")
    #threading.Thread(target=record_audio_and_process, daemon=True).start()

    print("All threads started successfully. Program is running.")

    # Keep the program running
    try:
        while True:
            time.sleep(1)  # Keeps the main program alive
    except KeyboardInterrupt:
        print("Program interrupted. Exiting...")


if __name__ == "__main__":
    main()
