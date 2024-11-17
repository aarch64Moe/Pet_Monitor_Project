import smbus2
import time
import numpy as np
from scipy.signal import find_peaks
from filters import butter_bandpass_filter, apply_median_filter
import shared_data

# Define I2C bus and MAX30102 I2C address
MAX30102_ADDR = 0x57
REG_FIFO_DATA = 0x07
MODE_CONFIGURATION = 0x09
SPO2_CONFIGURATION = 0x0A
LED_CONFIGURATION_RED = 0x0C
LED_CONFIGURATION_IR = 0x0D

# Initialize the I2C bus
bus = smbus2.SMBus(shared_data.bus_number)

# Threshold for minimum IR signal to detect a finger
IR_THRESHOLD = 50000

# Function to write to a register
def write_register(reg_addr, data):
    bus.write_byte_data(MAX30102_ADDR, reg_addr, data)

def initialize_sensor():
    # Reset sensor
    write_register(MODE_CONFIGURATION, 0x40)  # Reset bit set to 1 to reset the sensor
    time.sleep(1)

    # Configure sensor for SpO2 and HR mode
    write_register(MODE_CONFIGURATION, 0x03)  # SpO2 and HR mode

    # Configure SpO2 settings with higher sample rate and ADC range
    write_register(SPO2_CONFIGURATION, 0x3F)  # Sample rate 800 Hz, ADC range 4096

    # Set LED current for Red and IR LEDs to maximum
    write_register(LED_CONFIGURATION_RED, 0x7F)  # Maximum Red LED current
    write_register(LED_CONFIGURATION_IR, 0x7F)  # Maximum IR LED current

# Read data from the FIFO
def read_max30102(duration=10):  # Increase to 10 seconds for better accuracy
    red_data = []
    ir_data = []
    start_time = time.time()

    while time.time() - start_time < duration:
        # Read 6 bytes of data from the FIFO data register
        data = bus.read_i2c_block_data(MAX30102_ADDR, REG_FIFO_DATA, 6)

        # Convert the red and IR data to 18-bit values
        red = (data[0] << 16) | (data[1] << 8) | data[2]
        ir = (data[3] << 16) | (data[4] << 8) | data[5]

        # Only keep the lower 18 bits of each value
        red = red & 0x03FFFF
        ir = ir & 0x03FFFF

        # Store data
        red_data.append(red)
        ir_data.append(ir)
        time.sleep(0.1)

    return red_data, ir_data

def calculate_heart_rate(red_data, fs=10):
    # Apply a band-pass filter with adjusted cut-off frequencies
    filtered_data = butter_bandpass_filter(red_data, 1.0, 2.5, fs, order=3)

    # Apply a median filter to reduce noise
    filtered_data = apply_median_filter(filtered_data, kernel_size=5)

    # Use peak detection with stricter conditions
    peaks, _ = find_peaks(filtered_data, distance=fs * 0.6, prominence=0.5)

    # Calculate BPM by averaging intervals over the last 15 seconds
    if len(peaks) > 1:
        intervals = np.diff(peaks) / fs  # Convert to seconds
        avg_interval = np.mean(intervals)
        bpm = 60 / avg_interval
        return bpm

    return None

# Moving average calculation
def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Outlier rejection based on the previous average
def reject_outliers(heart_rate, previous_avg, threshold=10):
    if abs(heart_rate - previous_avg) > threshold:
        return previous_avg  # Keep the previous average if the change is too large
    return heart_rate




