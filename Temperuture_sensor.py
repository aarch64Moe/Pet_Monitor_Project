from smbus2 import SMBus
import shared_data 

import time


# Temperature and Humidity sensor
AHT21_ADDRESS = 0x38



def reset_sensor(bus):
    try:
        bus.write_byte(AHT21_ADDRESS, 0xBA)
        time.sleep(0.1)
        #print("Sensor reset.")
    except Exception as e:
        print(f"Error resetting sensor: {e}")

def start_measurement(bus):
    try:
        # Send the measurement command: 0xAC, followed by 0x33, 0x00
        bus.write_i2c_block_data(AHT21_ADDRESS, 0xAC, [0x33, 0x00])
        #print("Measurement command sent.")
        time.sleep(0.1)  # Wait for 100ms as per the datasheet before checking status
    except Exception as e:
        print(f"Error starting measurement: {e}")

def check_status(bus):
    try:
        status = bus.read_byte_data(AHT21_ADDRESS, 0x71)
        #print(f"Sensor status: {status:08b}")
        return status
    except Exception as e:
        print(f"Error reading sensor status: {e}")
        return None

def wait_for_ready_status(bus, timeout=5):
    start_time = time.time()
    while time.time() - start_time < timeout:
        status = check_status(bus)
        if status is not None and (status & 0x80) == 0:  # Bit 7 being 0 means the sensor is ready
            return True
        time.sleep(0.1)
    print("Sensor did not become ready in time.")
    return False

def read_temperature_humidity():
    with SMBus(shared_data.bus_number) as bus:
        try:
            reset_sensor(bus)
            time.sleep(0.1)  # Allow sensor to reset properly

            start_measurement(bus)
            time.sleep(0.08)  # Wait for 80ms before checking the status

            if wait_for_ready_status(bus):
                data = bus.read_i2c_block_data(AHT21_ADDRESS, 0x00, 6)
                #print(f"Raw data: {data}")

                # Calculate temperature and humidity
                raw_temperature = ((data[3] & 0x0F) << 16) | (data[4] << 8) | data[5]
                temperature = (raw_temperature * 200 / 1048576) - 50

                raw_humidity = (data[1] << 12) | (data[2] << 4) | (data[3] >> 4)
                humidity = raw_humidity * 100 / 1048576

                return temperature, humidity
            else:
                print("Sensor not ready, skipping read.")
                return None, None
        except Exception as e:
            print(f"Error reading from AHT21 sensor: {e}")
            return None, None


