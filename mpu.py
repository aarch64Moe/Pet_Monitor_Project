
import smbus2
import math
from collections import deque
import threading
import shared_data 
import numpy as np
import time

# Constants
BUFFER_SIZE = 3
ERROR_MARGIN = 0.5
FLUCTUATION_THRESHOLD = 0.01
RECORDING_DURATION = 1  # Set to 1 second for real-time

# MPU6050 Registers and their Addresses
MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B
GYRO_XOUT_H = 0x43


# Initialize I2C (SMBus)
bus = smbus2.SMBus(shared_data.bus_number)  # Use I2C bus 4 for Radxa Zero

# Wake up MPU6050
bus.write_byte_data(MPU6050_ADDR, PWR_MGMT_1, 0)

# Buffers
accel_buffer = {axis: deque(maxlen=BUFFER_SIZE) for axis in "xyz"}
gyro_buffer = {axis: deque(maxlen=BUFFER_SIZE) for axis in "xyz"}
pitch_buffer = deque(maxlen=BUFFER_SIZE)
roll_buffer = deque(maxlen=BUFFER_SIZE)

# Helper functions
def read_raw_data(addr):
    high = bus.read_byte_data(MPU6050_ADDR, addr)
    low = bus.read_byte_data(MPU6050_ADDR, addr + 1)
    value = (high << 8) | low
    return value - 65536 if value > 32768 else value

def moving_average(buffer, new_value):
    buffer.append(new_value)
    return sum(buffer) / len(buffer)


def get_pitch_roll(accel_x, accel_y, accel_z):
    pitch = math.atan2(accel_y, math.sqrt(accel_x ** 2 + accel_z ** 2)) * 180 / math.pi
    roll = math.atan2(-accel_x, accel_z) * 180 / math.pi
    return pitch, roll
    

def calculate_fluctuation(buffer):
    return max(0, max(buffer) - min(buffer) - ERROR_MARGIN)


def detect_movement():
    pitch_fluct = calculate_fluctuation(pitch_buffer)
    roll_fluct = calculate_fluctuation(roll_buffer)

    if pitch_fluct > FLUCTUATION_THRESHOLD and roll_fluct > FLUCTUATION_THRESHOLD:
        shared_data.walking_count += 1

        if shared_data.walking_count > 2:
            #shared_data.walking_count = 0
            shared_data.current_movement = "Moving"
            shared_data.sitting_count = 0
        return "Moving"

    else:
        shared_data.current_movement = "Standing Still"
        shared_data.sitting_count += 1
    return "Standing Still"




# Function to retrieve MPU6050 data with calibration applied
def get_mpu6050_data(accel_offsets, gyro_offsets, prev_data=None):
    # Read raw accelerometer and gyroscope data, subtracting the offsets
    accel_x = read_raw_data(ACCEL_XOUT_H) - accel_offsets["x"]
    accel_y = read_raw_data(ACCEL_XOUT_H + 2) - accel_offsets["y"]
    accel_z = read_raw_data(ACCEL_XOUT_H + 4) - accel_offsets["z"]

    gyro_x = read_raw_data(GYRO_XOUT_H) - gyro_offsets["x"]
    gyro_y = read_raw_data(GYRO_XOUT_H + 2) - gyro_offsets["y"]
    gyro_z = read_raw_data(GYRO_XOUT_H + 4) - gyro_offsets["z"]

    # Debug: Print calibrated accelerometer and gyroscope data
    # print(f"Calibrated Accelerometer: X={accel_x}, Y={accel_y}, Z={accel_z}")
    # print(f"Calibrated Gyroscope: X={gyro_x}, Y={gyro_y}, Z={gyro_z}")

    # Scale the raw data to proper units (accelerometer: g, gyroscope: degrees/sec)
    accel_x_scaled = accel_x / 16384.0
    accel_y_scaled = accel_y / 16384.0
    accel_z_scaled = accel_z / 16384.0
    gyro_x_scaled = gyro_x / 131.0
    gyro_y_scaled = gyro_y / 131.0
    gyro_z_scaled = gyro_z / 131.0

    # Debug: Print scaled accelerometer and gyroscope data
    # print(f"Scaled Accelerometer: X={accel_x_scaled}, Y={accel_y_scaled}, Z={accel_z_scaled}")
    # print(f"Scaled Gyroscope: X={gyro_x_scaled}, Y={gyro_y_scaled}, Z={gyro_z_scaled}")

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

