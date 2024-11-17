import sounddevice as sd
import numpy as np

sample_rate = 16000
duration = 5  # seconds
device_index = 1  # Replace with your device index

def audio_callback(indata, frames, time, status):
    if status:
        print("Status Error:", status)
    print("Captured audio data:", indata[:10])

print("Starting test...")
with sd.InputStream(device=device_index, callback=audio_callback, channels=1, samplerate=sample_rate):
    sd.sleep(duration * 1000)
print("Test finished.")
