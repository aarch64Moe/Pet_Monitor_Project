import smbus2
import time
import math
from collections import deque
import threading
import subprocess
import soundfile as sf
import numpy as np
import os
from datetime import datetime  # Added for timestamped filenames


# MPU6050 Registers and their Addresses
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43

# Initialize I2C (SMBus)
bus = smbus2.SMBus(4)  # Use I2C bus 1 for Radxa Zero

# Wake up MPU6050 (it starts in sleep mode)
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

# Moving average buffer size
BUFFER_SIZE = 10
FLUCTUATION_THRESHOLD = 0.1  # Threshold for pitch/roll fluctuation to detect walking
LAYING_THRESHOLD_TIME = 2  # Time in seconds to detect laying down when stable
ERROR_MARGIN = 0.5  # Error margin for sensor noise

# Variables to track stability for laying down detection
stability_counter = 0
stability_start_time = None
walking_count = 0  # Counter for consecutive walking detections
recording_flag = False
lock = threading.Lock()

# Buffers for accelerometer and gyroscope data
accel_buffer = {"x": deque(maxlen=BUFFER_SIZE), "y": deque(maxlen=BUFFER_SIZE), "z": deque(maxlen=BUFFER_SIZE)}
gyro_buffer = {"x": deque(maxlen=BUFFER_SIZE), "y": deque(maxlen=BUFFER_SIZE), "z": deque(maxlen=BUFFER_SIZE)}

# Buffers for pitch and roll
pitch_buffer = deque(maxlen=BUFFER_SIZE)
roll_buffer = deque(maxlen=BUFFER_SIZE)

# Function to retrieve MPU6050 data
def get_mpu6050_data(accel_offsets, gyro_offsets, prev_data=None):
    # Read raw accelerometer and gyroscope data, subtracting the offsets
    accel_x = read_raw_data(ACCEL_XOUT_H) - accel_offsets["x"]
    accel_y = read_raw_data(ACCEL_XOUT_H + 2) - accel_offsets["y"]
    accel_z = read_raw_data(ACCEL_XOUT_H + 4) - accel_offsets["z"]

    gyro_x = read_raw_data(GYRO_XOUT_H) - gyro_offsets["x"]
    gyro_y = read_raw_data(GYRO_XOUT_H + 2) - gyro_offsets["y"]
    gyro_z = read_raw_data(GYRO_XOUT_H + 4) - gyro_offsets["z"]

    # Scale the raw data to proper units (accelerometer: g, gyroscope: degrees/sec)
    accel_x_scaled = accel_x / 16384.0
    accel_y_scaled = accel_y / 16384.0
    accel_z_scaled = accel_z / 16384.0
    gyro_x_scaled = gyro_x / 131.0
    gyro_y_scaled = gyro_y / 131.0
    gyro_z_scaled = gyro_z / 131.0

    # If previous data exists, apply a high-pass filter to the gyroscope data
    if prev_data:
        gyro_x_scaled = high_pass_filter(gyro_x_scaled, prev_data["gyro"]["x"])
        gyro_y_scaled = high_pass_filter(gyro_y_scaled, prev_data["gyro"]["y"])
        gyro_z_scaled = high_pass_filter(gyro_z_scaled, prev_data["gyro"]["z"])

    # Apply moving average filter to smooth data
    accel_x_avg = moving_average(accel_buffer["x"], accel_x_scaled)
    accel_y_avg = moving_average(accel_buffer["y"], accel_y_scaled)
    accel_z_avg = moving_average(accel_buffer["z"], accel_z_scaled)
    gyro_x_avg = moving_average(gyro_buffer["x"], gyro_x_scaled)
    gyro_y_avg = moving_average(gyro_buffer["y"], gyro_y_scaled)
    gyro_z_avg = moving_average(gyro_buffer["z"], gyro_z_scaled)

    # Calculate pitch and roll based on accelerometer data
    pitch, roll = get_pitch_roll(accel_x_avg, accel_y_avg, accel_z_avg)

    # Update the pitch and roll buffers for fluctuation detection
    pitch_buffer.append(pitch)
    roll_buffer.append(roll)

    return {
        "accel": {"x": accel_x_avg, "y": accel_y_avg, "z": accel_z_avg},
        "gyro": {"x": gyro_x_avg, "y": gyro_y_avg, "z": gyro_z_avg},
        "pitch": pitch,
        "roll": roll
    }

# Function to interactively calibrate the sensor
def interactive_calibration():
    print("Starting interactive calibration process...")

    positions = {
        "flat": "Place the sensor flat on a stable surface and press Enter.",
        "on its left side": "Place the sensor on its left side and press Enter.",
        "on its right side": "Place the sensor on its right side and press Enter.",
        "face up": "Place the sensor face up and press Enter.",
        "face down": "Place the sensor face down and press Enter."
    }

    accel_offsets = {"x": 0, "y": 0, "z": 0}
    gyro_offsets = {"x": 0, "y": 0, "z": 0}

    for position, instruction in positions.items():
        input(instruction)
        print(f"Gathering data for {position}...")
        x_offset, y_offset, z_offset = 0, 0, 0
        gyro_x_offset, gyro_y_offset, gyro_z_offset = 0, 0, 0

        for _ in range(200):
            accel_x = read_raw_data(ACCEL_XOUT_H)
            accel_y = read_raw_data(ACCEL_XOUT_H + 2)
            accel_z = read_raw_data(ACCEL_XOUT_H + 4)
            gyro_x = read_raw_data(GYRO_XOUT_H)
            gyro_y = read_raw_data(GYRO_XOUT_H + 2)
            gyro_z = read_raw_data(GYRO_XOUT_H + 4)

            x_offset += accel_x
            y_offset += accel_y
            z_offset += accel_z
            gyro_x_offset += gyro_x
            gyro_y_offset += gyro_y
            gyro_z_offset += gyro_z
            time.sleep(0.01)

        accel_offsets["x"] += x_offset / 200
        accel_offsets["y"] += y_offset / 200
        accel_offsets["z"] += z_offset / 200

        gyro_offsets["x"] += gyro_x_offset / 200
        gyro_offsets["y"] += gyro_y_offset / 200
        gyro_offsets["z"] += gyro_z_offset / 200

        print(f"Data gathered for {position}.")

    accel_offsets = {key: val / len(positions) for key, val in accel_offsets.items()}
    gyro_offsets = {key: val / len(positions) for key, val in gyro_offsets.items()}

    print("Interactive calibration complete.")
    return accel_offsets, gyro_offsets


# Function to read raw data from a register
def read_raw_data(addr):
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    value = (high << 8) | low
    if value > 32768:
        value = value - 65536
    return value


# High-pass filter for gyroscope data to focus on significant rotations
def high_pass_filter(new_value, prev_filtered_value, alpha=0.95):
    return alpha * (prev_filtered_value + new_value)


# Moving average filter to smooth data
def moving_average(buffer, new_value):
    buffer.append(new_value)
    return sum(buffer) / len(buffer)


# Calculate the magnitude of the acceleration vector
def calculate_magnitude(accel_x, accel_y, accel_z):
    return math.sqrt(accel_x ** 2 + accel_y ** 2 + accel_z ** 2)


# Calculate fluctuation of pitch/roll within the buffer with error margin
def calculate_fluctuation(buffer):
    if len(buffer) < 2:
        return 0
    fluctuation = max(buffer) - min(buffer)
    return max(0, fluctuation - ERROR_MARGIN)



# Detect movement based on pitch/roll fluctuations
def detect_movement(pitch_buffer, roll_buffer):
    global stability_counter, stability_start_time, walking_count, recording_flag, last_recording_time

    pitch_fluctuation = calculate_fluctuation(pitch_buffer)
    roll_fluctuation = calculate_fluctuation(roll_buffer)

    # Detect walking if fluctuations are small (within threshold)
    if pitch_fluctuation > FLUCTUATION_THRESHOLD and roll_fluctuation > FLUCTUATION_THRESHOLD:
        stability_counter = 0
        stability_start_time = None
        with lock:
            walking_count += 1
            # Check if the walking count is 5 and if we are ready to record again
            if walking_count >= 10 and not recording_flag:
                current_time = time.time()
                # Ensure 10 seconds have passed since the last recording
                if current_time - last_recording_time >= 10:
                    print("Walking detected 10 times, starting recording...")
                    recording_flag = True
                    threading.Thread(target=record_audio_and_process).start()
        return "Walking"

    # If fluctuations are bigger, start tracking stability for laying down
    if pitch_fluctuation <= FLUCTUATION_THRESHOLD or roll_fluctuation <= FLUCTUATION_THRESHOLD:
        if stability_start_time is None:
            stability_start_time = time.time()
        else:
            if time.time() - stability_start_time > LAYING_THRESHOLD_TIME:
                return "Laying Down"

    walking_count = 0
    # If stability not achieved, reset the counter
    stability_counter = 0
    stability_start_time = None
    return "Sitting/Standing Still"


# Detect rotation using gyroscope data
def detect_rotation(gyro):
    if abs(gyro["x"]) > 150 or abs(gyro["y"]) > 150:
        return "Shaking/Quick Movement"
    elif abs(gyro["z"]) > 80:
        return "Turning/Changing Direction"
    else:
        return "Calm/Steady"


# Get pitch and roll from accelerometer data
def get_pitch_roll(accel_x, accel_y, accel_z):
    pitch = math.atan2(accel_y, math.sqrt(accel_x ** 2 + accel_z ** 2)) * 180.0 / math.pi
    roll = math.atan2(-accel_x, accel_z) * 180.0 / math.pi
    return pitch, roll


# Audio processing functions
def apply_highpass_filter(data, samplerate):
    # Implement high-pass filter logic
    return data


def apply_lowpass_filter(data, samplerate):
    # Implement low-pass filter logic
    return data


def amplify_audio(data, db_gain=30):
    factor = 10 ** (db_gain / 20)
    return data * factor


def noise_reduction(data):
    # Implement noise reduction logic
    return data



# Global variable to track the last recording time
last_recording_time = 0  # Initialize to zero

# Record and process audio
def record_audio_and_process():
    global walking_count, recording_flag, last_recording_time

    # Create a unique filename using a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_audio_filename = f"detected_voice_{timestamp}.wav"
    processed_audio_filename = f"processed_voice_{timestamp}.wav"

    # Set the duration of the recording (e.g., 5 seconds)
    duration = 1  # Record for 5 seconds

    # Command for recording the audio (with a 5-second duration)
    command = ["arecord", "-D", "plughw:0,0", "-c1", "-r", "48000", "-f", "S32_LE", "-t", "wav", "--buffer-size=192000", "-d", str(duration), raw_audio_filename]
    subprocess.run(command)
    print(f"Recording completed: {raw_audio_filename}. Applying audio processing...")

    # Check if the file was successfully created
    if os.path.exists(raw_audio_filename):
        # Load the recorded audio
        audio_data, samplerate = sf.read(raw_audio_filename)

        # Apply audio filters
        #audio_data = apply_highpass_filter(audio_data, samplerate)
        #audio_data = apply_lowpass_filter(audio_data, samplerate)
        audio_data = amplify_audio(audio_data, 30)
        audio_data = noise_reduction(audio_data)

        # Save the processed audio
        sf.write(processed_audio_filename, audio_data, samplerate)
        print(f"Processed audio saved: {processed_audio_filename}")
    else:
        print(f"Error: {raw_audio_filename} not found!")

    # Reset the walking count and recording flag after recording
    with lock:
        walking_count = 0
        recording_flag = False
        last_recording_time = time.time()  # Update the last recording time

def detect_movement_thread(mpu_data, pitch_buffer, roll_buffer):
    movement = detect_movement(pitch_buffer, roll_buffer)
    print(f"Movement: {movement}")

# Main function
def main():
    accel_offsets, gyro_offsets = interactive_calibration()
    prev_data = None
    while True:
        start_time = time.time()
        mpu_data = get_mpu6050_data(accel_offsets, gyro_offsets, prev_data)
        prev_data = mpu_data

        movment_thread = threading.Thread(target =detect_movement_thread, args=(mpu_data, pitch_buffer, roll_buffer))
        movment_thread.start()

        #movement = detect_movement(pitch_buffer, roll_buffer)

        # Detect movement and rotation
        accel_magnitude = calculate_magnitude(mpu_data["accel"]["x"], mpu_data["accel"]["y"], mpu_data["accel"]["z"])

        rotation = detect_rotation(mpu_data["gyro"])

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


        print(f"Rotation: {rotation}")
        print(f"Posture: {posture}")
        print("=" * 40)

        movment_thread.join()

        # Sleep to maintain a consistent sampling rate
        time.sleep(0.1)


if __name__ == "__main__":
    main()
