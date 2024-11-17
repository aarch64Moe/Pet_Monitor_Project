#!/bin/bash

# Stop pulseaudio
pulseaudio --kill
echo "Pulseaudio stopped."

# Wait a moment to ensure pulseaudio has stopped
sleep 1



# Restart pulseaudio after the program finishes
pulseaudio --start
echo "Pulseaudio restarted."

# Run your Python classification script
python3 /home/radxa/yamnet/test.py


