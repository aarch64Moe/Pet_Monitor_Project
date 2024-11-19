import serial
import threading
import time

# Global variable to store the latest GPS coordinates
current_coordinates = {"latitude": None, "longitude": None}
gps_running = False  # Control flag for GPS tracking



def convert_to_decimal_degrees(raw_value, direction):
    """Convert raw NMEA format to decimal degrees."""
    degrees = int(raw_value // 100)
    minutes = raw_value % 100
    decimal_degrees = degrees + (minutes / 60)
    if direction in ['S', 'W']:
        decimal_degrees *= -1
    return decimal_degrees

def parse_gps_data(data):
    """Parse NMEA sentences to extract latitude and longitude."""
    global current_coordinates
    if data.startswith('$GPGGA'):
        parts = data.split(',')
        if len(parts) > 5:
            try:
                # Parse latitude
                raw_lat = float(parts[2]) if parts[2] else 0
                lat_dir = parts[3]
                latitude = convert_to_decimal_degrees(raw_lat, lat_dir)

                # Parse longitude
                raw_lon = float(parts[4]) if parts[4] else 0
                lon_dir = parts[5]
                longitude = convert_to_decimal_degrees(raw_lon, lon_dir)

                current_coordinates["latitude"] = latitude
                current_coordinates["longitude"] = longitude
                #print(f"Updated coordinates: {current_coordinates}")

            except ValueError:
                print("Failed to parse GPS data")
        else:
            print("Incomplete GPS data")

gps_lock = threading.Lock()

def start_gps_tracking(port="/dev/ttyS2", baudrate=9600, retries=3, retry_delay=5):
    global gps_running
    with gps_lock:  # Ensure only one thread can access this block
        if gps_running:
            print("GPS tracking is already running.")
            return

        gps_running = True
        attempt = 0

        while gps_running and attempt < retries:
            try:
                with serial.Serial(port, baudrate=baudrate, timeout=1) as gps_serial:
                    print(f"Connected to GPS device on {port}.")
                    while gps_running:
                        line = gps_serial.readline().decode('ascii', errors='replace').strip()
                        if line:
                            parse_gps_data(line)
                        else:
                            print("No data received, retrying in 2 seconds...")
                            time.sleep(2)
            except serial.SerialException as e:
                print(f"Error accessing serial port: {e}. Retrying in {retry_delay} seconds...")
                attempt += 1
                time.sleep(retry_delay)
            except Exception as e:
                print(f"Unexpected error: {e}")
                break

        if attempt >= retries:
            print("Failed to connect to GPS device after multiple attempts.")
        else:
            print("GPS tracking stopped.")



def stop_gps_tracking():
    """Stop GPS tracking loop."""
    global gps_running
    if gps_running:
        gps_running = False
        print("GPS tracking has been stopped.")
    else:
        print("GPS tracking was already stopped.")

def get_current_coordinates():
    """Return the latest GPS coordinates."""
    return current_coordinates
