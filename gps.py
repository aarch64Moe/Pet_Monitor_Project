import math
import time
import serial

# Constants
AVERAGE_STEP_LENGTH = 0.35
distance_traveled = 0
MOVEMENT_DISTANCE_THRESHOLD = 5.0
MIN_TIME_INTERVAL = 3
MINIMUM_SPEED_THRESHOLD = 0.5
MAXIMUM_SPEED_THRESHOLD = 5.0  # Maximum realistic speed for a pet in m/s

# Tracking variables
speed_buffer = []
previous_lat = None
previous_lon = None
previous_time = None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371e3
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_speed(lat1, lon1, lat2, lon2, time_elapsed):
    distance = haversine(lat1, lon1, lat2, lon2)
    return distance / time_elapsed if time_elapsed > 0 else 0

gps_serial = serial.Serial('/dev/ttyS2', baudrate=9600, timeout=1)
print("Reading GPS data from /dev/ttyS2...")

def parse_gps_data(data):
    global previous_lat, previous_lon, previous_time, distance_traveled, speed_buffer

    if data.startswith('$GPGGA'):
        parts = data.split(',')
        if len(parts) > 8:
            # Get latitude, longitude, and HDOP
            lat = float(parts[2])
            lon = float(parts[4])
            hdop = float(parts[8])

            # Filter based on HDOP quality (ignore high HDOP values)
            if hdop > 6.0:
                print("Poor GPS signal quality (HDOP too high), ignoring reading.")
                return

            current_time = time.time()

            if previous_lat is not None and previous_lon is not None and previous_time is not None:
                time_elapsed = current_time - previous_time

                # Skip reading if time interval is too short
                if time_elapsed < MIN_TIME_INTERVAL:
                    return

                distance = haversine(previous_lat, previous_lon, lat, lon)

                # Ignore small movements due to noise
                if distance < MOVEMENT_DISTANCE_THRESHOLD:
                    speed = 0.0
                else:
                    speed = calculate_speed(previous_lat, previous_lon, lat, lon, time_elapsed)

                # Filter based on maximum realistic speed
                if speed > MAXIMUM_SPEED_THRESHOLD:
                    print(f"Unrealistic speed detected ({speed:.2f} m/s), ignoring reading.")
                    speed = 0.0

                # Add speed to buffer and calculate average
                speed_buffer.append(speed)
                if len(speed_buffer) > 5:  # Keep last 5 readings
                    speed_buffer.pop(0)
                average_speed = sum(speed_buffer) / len(speed_buffer)

                # Determine activity based on average speed
                if average_speed < MINIMUM_SPEED_THRESHOLD:
                    activity = "Resting"
                elif average_speed < 1.5:
                    activity = "Walking"
                    distance_traveled += distance
                else:
                    activity = "Running"

                print(f"Average Speed: {average_speed:.2f} m/s, Activity: {activity}")

            # Update previous values
            previous_lat, previous_lon, previous_time = lat, lon, current_time

try:
    while True:
        line = gps_serial.readline().decode('ascii', errors='replace').strip()
        if line:
            parse_gps_data(line)
        time.sleep(.1)
except KeyboardInterrupt:
    print("Stopping GPS tracking...")
finally:
    gps_serial.close()
