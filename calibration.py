import os
import json
from mpu import read_raw_data

CALIBRATION_FILE = "calibration_data.json"

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
        input(instruction)  # User places sensor in the required position
        print(f"Gathering data for {position}...")

        # Accumulate raw data for each position
        accel_x_sum, accel_y_sum, accel_z_sum = 0, 0, 0
        gyro_x_sum, gyro_y_sum, gyro_z_sum = 0, 0, 0

        # Gather 200 samples to average out sensor noise
        for _ in range(200):
            accel_x = read_raw_data(ACCEL_XOUT_H)
            accel_y = read_raw_data(ACCEL_XOUT_H + 2)
            accel_z = read_raw_data(ACCEL_XOUT_H + 4)
            gyro_x = read_raw_data(GYRO_XOUT_H)
            gyro_y = read_raw_data(GYRO_XOUT_H + 2)
            gyro_z = read_raw_data(GYRO_XOUT_H + 4)

            # Accumulate data
            accel_x_sum += accel_x
            accel_y_sum += accel_y
            accel_z_sum += accel_z
            gyro_x_sum += gyro_x
            gyro_y_sum += gyro_y
            gyro_z_sum += gyro_z

            time.sleep(0.01)  # Small delay to ensure accurate readings

        # Compute the average offset over 200 samples
        accel_offsets["x"] += accel_x_sum / 200
        accel_offsets["y"] += accel_y_sum / 200
        accel_offsets["z"] += accel_z_sum / 200
        gyro_offsets["x"] += gyro_x_sum / 200
        gyro_offsets["y"] += gyro_y_sum / 200
        gyro_offsets["z"] += gyro_z_sum / 200

        # Print gathered data for each position
        print(f"Data gathered for {position}. Accel Offsets: {accel_offsets}, Gyro Offsets: {gyro_offsets}")

    # Normalize the final offsets by dividing by the number of positions
    accel_offsets = {key: val / len(positions) for key, val in accel_offsets.items()}
    gyro_offsets = {key: val / len(positions) for key, val in gyro_offsets.items()}

    # Debug: Print final calibration offsets
    print(f"Final Accel Offsets: {accel_offsets}")
    print(f"Final Gyro Offsets: {gyro_offsets}")

    return accel_offsets, gyro_offsets


# Save calibration data to a file
def save_calibration_data(accel_offsets, gyro_offsets):
    calibration_data = {
        "accel_offsets": accel_offsets,
        "gyro_offsets": gyro_offsets
    }
    with open(CALIBRATION_FILE, 'w') as file:
        json.dump(calibration_data, file)
    print("Calibration data saved to file.")


# Load calibration data from a file
def load_calibration_data():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, 'r') as file:
            calibration_data = json.load(file)
        print("Calibration data loaded from file.")
        return calibration_data["accel_offsets"], calibration_data["gyro_offsets"]
    else:
        return None, None


# Function to get calibration data
def get_calibration_data():
    accel_offsets, gyro_offsets = load_calibration_data()

    # If calibration data not found, perform interactive calibration
    if accel_offsets is None or gyro_offsets is None:
        print("No calibration data found, performing interactive calibration.")
        accel_offsets, gyro_offsets = interactive_calibration()
        save_calibration_data(accel_offsets, gyro_offsets)

    return accel_offsets, gyro_offsets






