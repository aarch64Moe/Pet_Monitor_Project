import gpiod
import time

# Define GPIO chip and pins
chip = gpiod.Chip('gpiochip0')  # Change to the appropriate GPIO chip
lines = [chip.get_line(pin) for pin in [17, 18, 27]]  # Replace with GPIO line numbers

# Configure GPIO lines as outputs
for line in lines:
    config = gpiod.LineRequest()
    config.consumer = "gpio-control"
    config.request_type = gpiod.LINE_REQ_DIR_OUT
    line.request(config)

def disable_pins():
    """Disable GPIO pins (turn off peripherals)."""
    print("Disabling GPIO pins...")
    for line in lines:
        line.set_value(0)  # Set GPIO line to LOW

def enable_pins():
    """Enable GPIO pins (turn on peripherals)."""
    print("Enabling GPIO pins...")
    for line in lines:
        line.set_value(1)  # Set GPIO line to HIGH


