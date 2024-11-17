import serial
import time
import math

# Constants
SERIAL_PORT = '/dev/ttyS2'
BAUD_RATE = 9600
HDOP_THRESHOLD = 5.0
MOVEMENT_DISTANCE_THRESHOLD = 0.5  # Ignore small movements below 5 meters
MINIMUM_SPEED_THRESHOLD = 0.1
MAXIMUM_SPEED_THRESHOLD = 5.0
AVERAGE_WINDOW = 5

# Initialize variables
previous_lat = None
previous_lon = None
previous_time = None
speed_buffer = []
distance_traveled = 0


def initialize_gps(serial_connection):
    """Configure GPS module for desired baud rate and update rate."""
    serial_connection.write(b'$PMTK220,1000*1F\r\n')  # Set update rate to 1 Hz
    serial_connection.write(b'$PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0*28\r\n')
    time.sleep(1)


def haversine(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance in meters between two points."""
    R = 6371e3  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def calculate_speed(lat1, lon1, lat2, lon2, time_elapsed):
    """Calculate speed in m/s based on the distance between two points."""
    distance = haversine(lat1, lon1, lat2, lon2)
    return distance / time_elapsed if time_elapsed > 0 else 0


def parse_gps_data(data):
    global previous_lat, previous_lon, previous_time, speed_buffer, distance_traveled

    hdop = None  # Initialize hdop to None as a default

    if data.startswith('$GPGGA') or data.startswith('$GPRMC'):
        parts = data.split(',')

        # Parse $GPGGA sentence
        if data.startswith('$GPGGA') and len(parts) > 8 and parts[2] and parts[4] and parts[8]:
            try:
                lat = float(parts[2]) / 100.0  # Latitude in degrees
                lon = float(parts[4]) / 100.0  # Longitude in degrees
                hdop = float(parts[8])  # Horizontal Dilution of Precision
            except ValueError:
                print("Invalid GPS data, skipping this reading.")
                return

        # Parse $GPRMC sentence
        elif data.startswith('$GPRMC') and len(parts) > 5 and parts[3] and parts[5]:
            try:
                lat = float(parts[3]) / 100.0
                lon = float(parts[5]) / 100.0
            except ValueError:
                print("Invalid GPS data, skipping this reading.")
                return
        else:
            print("Incomplete GPS data, skipping this reading.")
            return

        # Check HDOP only if it was set from a $GPGGA sentence
        if hdop is not None and hdop > HDOP_THRESHOLD:
            print("Poor GPS signal quality (HDOP too high), ignoring reading.")
            return

        current_time = time.time()
        if previous_lat is not None and previous_lon is not None and previous_time is not None:
            time_elapsed = current_time - previous_time
            distance = haversine(previous_lat, previous_lon, lat, lon)

            # Ignore small movements due to noise
            if distance < MOVEMENT_DISTANCE_THRESHOLD:
                speed = 0.0
            else:
                speed = calculate_speed(previous_lat, previous_lon, lat, lon, time_elapsed)

            # Filter unrealistic speeds
            if speed > MAXIMUM_SPEED_THRESHOLD:
                print(f"Unrealistic speed detected ({speed:.2f} m/s), ignoring reading.")
                speed = 0.0

            # Buffer and average speed readings
            speed_buffer.append(speed)
            if len(speed_buffer) > AVERAGE_WINDOW:
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

            print(
                f"Average Speed: {average_speed:.2f} m/s, Activity: {activity}, Distance Traveled: {distance_traveled:.2f} m")

        previous_lat, previous_lon, previous_time = lat, lon, current_time


# Serial connection setup
gps_serial = serial.Serial(SERIAL_PORT, baudrate=BAUD_RATE, timeout=1)
initialize_gps(gps_serial)

try:
    while True:
        line = gps_serial.readline().decode('ascii', errors='replace').strip()
        if line:
            #print(f"Raw GPS Data: {line}")
            parse_gps_data(line)
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping GPS tracking...")
finally:
    gps_serial.close()
