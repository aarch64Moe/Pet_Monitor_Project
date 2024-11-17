import smbus2
import time
import math
import queue
import threading
import numpy as np
from datetime import datetime
import sounddevice as sd
from scipy.signal import resample
import soundfile as sf  # Required for saving audio files
import tflite_runtime.interpreter as tflite

# Import the function from the temperature sensor module
import shared_data 
from Temperuture_sensor import read_temperature_humidity  # Import from the sensor file
from heart import initialize_sensor, read_max30102, monitor_heart_rate_when_still
from filters import apply_filter
from mpu import get_mpu6050_data,detect_movement
from calibration import get_calibration_data


# Global variables to store sensor readings
heart_rate = None
temperature = None
humidity = None
mpu_data = None

# Variables
RECORDING_DURATION = 1  # Set to 1 second for real-time
SAMPLERATE = 48000  # Sample rate for audio
CHANNELS = 1  # Mono audio


# Resample the audio to 16 kHz
def resample_audio(audio_data, orig_samplerate, target_samplerate=16000):
    number_of_samples = int(len(audio_data) * float(target_samplerate) / orig_samplerate)
    resampled_audio = resample(audio_data, number_of_samples)
    return resampled_audio


# Prepare audio for YAMNet: ensure it's mono, resampled to 16 kHz, and 1D
def prepare_audio_for_yamnet(audio_data, orig_samplerate):
    # Convert to mono (if stereo)
    if len(audio_data.shape) > 1 and audio_data.shape[1] == 2:  # Stereo to mono
        audio_data = np.mean(audio_data, axis=1)

    # Resample the audio to 16 kHz for YAMNet
    resampled_audio = resample_audio(audio_data, orig_samplerate, target_samplerate=16000)

    # Ensure the audio is in the shape (1, num_samples)
    return np.expand_dims(resampled_audio, axis=0).astype(np.float32)



# Function to load YAMNet labels from a CSV file
def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [line.strip().split(',')[2] for line in f.readlines()[1:]]
    return labels


# Function to load YAMNet model
def load_yamnet_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Classify audio with YAMNet
def classify_audio_with_yamnet(interpreter, audio_data=None):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Use dummy input if audio_data is not provided
    if audio_data is None:
        audio_data = np.array([0.0], dtype=np.float32)

    # Ensure audio_data is of expected shape and dtype
    audio_data = np.reshape(audio_data, input_details[0]['shape']).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], audio_data)
    interpreter.invoke()

    # Get the predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions[0]





# Amplify the audio in memory
def amplify_audio(data, db_gain=30):
    factor = 10 ** (db_gain / 20)
    return data * factor

def record_audio_and_process():
    global last_recording_time, walking_count, recording_flag
    duration = RECORDING_DURATION

    # Load YAMNet model (only do this once)
    yamnet_model = load_yamnet_model("/home/radxa/yamnet/yamnet.tflite")  # Update with correct model path
    labels = load_labels('yamnet_class_map.csv')  # Load the labels

    # Capture the start time before starting recording
    start_time = time.time()

    print(f"Recording audio for {duration} second(s)...")

    # Record audio in memory using sounddevice
    audio_data = sd.rec(int(duration * SAMPLERATE), samplerate=SAMPLERATE, channels=CHANNELS, dtype='float32')
    sd.wait()  # Wait until recording is done

    # Amplify the audio signal in memory
    audio_data = amplify_audio(audio_data, 40)

    # Apply the selected filter
    filtered_audio = apply_filter(audio_data.flatten(), 'butterworth', fs=SAMPLERATE)

    # Prepare the audio for YAMNet: Convert to 16 kHz, mono, and flatten to 1D
    yamnet_input_audio = prepare_audio_for_yamnet(filtered_audio, SAMPLERATE)

    # Classify the prepared audio with YAMNet
    yamnet_predictions = classify_audio_with_yamnet(yamnet_model, yamnet_input_audio)
    top_predictions = classify_sound(yamnet_predictions, labels)

    # Display the classification results
    print("YAMNet Predictions:")
    for label, score in top_predictions:
        print(f"{label}: {score:.4f}")

    # Save the processed audio
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_audio_filename = f"processed_voice_{timestamp}.wav"
    sf.write(processed_audio_filename, filtered_audio, SAMPLERATE)

    # Record processing time  
    end_time = time.time()
    processing_time = end_time - start_time



    # Reset flags
    with shared_data.lock:
        shared_data.walking_count = 0
        shared_data.recording_flag = False
        shared_data.last_recording_time = time.time()
        
    print(f"Processed audio saved: {processed_audio_filename}")
    print(f"Time taken from recording to saving: {processing_time:.2f} seconds")        



# Function to display temperature every 10 seconds
def temperature_thread():
    while True:
        temp, hum = read_temperature_humidity()
        if temp is not None and hum is not None:
            print(f"Temperature: {temp:.2f} °C, Humidity: {hum:.2f} %")                    
        else:
            print("Failed to read temperature and humidity.")
        time.sleep(10)  # Display every 10 seconds

# Main thread function to handle MPU6050 data
def mpu6050_thread(accel_offsets, gyro_offsets, prev_data):
    global mpu_data
    while True:
        mpu_data = get_mpu6050_data(accel_offsets, gyro_offsets, prev_data)
        prev_data = mpu_data
        time.sleep(0.1)


# Main function
def main():

    
    accel_offsets, gyro_offsets = get_calibration_data()
    
    # Start the mpu thread
    threading.Thread(target=mpu6050_thread, args=(accel_offsets, gyro_offsets, None), daemon=True).start()

    # Start the temperature thread
    threading.Thread(target=temperature_thread, daemon=True).start()

    # Start the heart rate monitoring thread
    threading.Thread(target=monitor_heart_rate_when_still,  daemon=True).start()
    
   # Start the recording and proccessing thread in mpu.py 
    while True:
        if mpu_data is not None:
            shared_data.movement = detect_movement()

            if shared_data.movement == "Walking":
                print(f"Movement: {shared_data.movement}")
        time.sleep(0.1)
"""
                # Interpret posture based on pitch and roll
                pitch = mpu_data["pitch"]
                roll = mpu_data["roll"]
                if pitch > 30:
                    posture = "Head Up (Looking Up)"
                elif pitch < -30:
                    posture = "Head Down (Looking Down)"
                elif roll > 30:
                    posture = "Leaning Right/Lying on Right Side"
                elif roll < -30:
                    posture = "Leaning Left/Lying on Left Side"
                else:
                    posture = "Upright/Neutral Position"
                print(f"Posture: {posture}")
"""






if __name__ == "__main__":
    main()
