import serial
import time
import math

# GPS configuration
SERIAL_PORT = '/dev/ttyS2'
BAUD_RATE = 9600
HDOP_THRESHOLD = 3.0
MOVEMENT_DISTANCE_THRESHOLD = 1.0
MINIMUM_SPEED_THRESHOLD = 0.01
MAXIMUM_SPEED_THRESHOLD = 5.0

# Initialize variables
block_duration = 5  # seconds
previous_block = []
current_block = []
previous_lat = None
previous_lon = None
previous_time = None


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


def process_block(block):
    """Calculate the average speed and total distance for a 5-second block."""
    total_distance = 0
    total_time = 0
    speeds = []

    for i in range(1, len(block)):
        lat1, lon1, t1 = block[i - 1]
        lat2, lon2, t2 = block[i]
        time_elapsed = t2 - t1

        distance = haversine(lat1, lon1, lat2, lon2)
        speed = calculate_speed(lat1, lon1, lat2, lon2, time_elapsed)

        if distance > MOVEMENT_DISTANCE_THRESHOLD:
            total_distance += distance
            speeds.append(speed)
            total_time += time_elapsed

    average_speed = sum(speeds) / len(speeds) if speeds else 0
    return average_speed, total_distance


def compare_blocks(previous_avg_speed, previous_distance, current_avg_speed, current_distance):
    """Compare two blocks to determine movement or activity."""
    if current_distance > previous_distance and current_distance > 5 and current_avg_speed > MINIMUM_SPEED_THRESHOLD:
        return "Moving faster"
    elif current_distance < previous_distance:
        return "Slowing down or resting"
    else:
        return "Consistent movement"


# Serial connection setup
gps_serial = serial.Serial(SERIAL_PORT, baudrate=BAUD_RATE, timeout=1)

try:
    while True:
        start_time = time.time()
        current_block = []

        # Collect data for 5 seconds
        while time.time() - start_time < block_duration:
            line = gps_serial.readline().decode('ascii', errors='replace').strip()
            if line.startswith('$GPGGA'):
                parts = line.split(',')
                if len(parts) > 8 and parts[2] and parts[4] and parts[8]:
                    try:
                        lat = float(parts[2]) / 100.0
                        lon = float(parts[4]) / 100.0
                        hdop = float(parts[8])

                        # Check HDOP for quality
                        if hdop <= HDOP_THRESHOLD:
                            current_time = time.time()
                            current_block.append((lat, lon, current_time))

                    except ValueError:
                        print("Invalid GPS data, skipping this reading.")

        # Process the current block
        if previous_block:
            prev_avg_speed, prev_distance = process_block(previous_block)
            curr_avg_speed, curr_distance = process_block(current_block)
            result = compare_blocks(prev_avg_speed, prev_distance, curr_avg_speed, curr_distance)
            print(f"Previous Block - Avg Speed: {prev_avg_speed:.2f} m/s, Distance: {prev_distance:.2f} m")
            print(f"Current Block - Avg Speed: {curr_avg_speed:.2f} m/s, Distance: {curr_distance:.2f} m")
            print(f"Comparison Result: {result}")

        # Update blocks
        previous_block = current_block

except KeyboardInterrupt:
    print("Stopping GPS tracking...")
finally:
    gps_serial.close()
