import serial
import time

# Set up the serial connection
gps_serial = serial.Serial('/dev/ttyS2', baudrate=9600, timeout=1)
print("Reading GPS data from /dev/ttyS2...")

try:
    while True:
        line = gps_serial.readline().decode('ascii', errors='replace').strip()
        print(f"Raw GPS Data: {line}")  # Print raw data to diagnose the issue

        # Add your parsing function if needed to process complete data
        # parse_gps_data(line)

        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping GPS tracking...")
finally:
    gps_serial.close()
