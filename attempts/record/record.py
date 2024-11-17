import os
import time
import numpy as np
import subprocess
import threading
from scipy.io import wavfile
from scipy.signal import butter, lfilter

# Parameters
RECORDING_DURATION = 5  # seconds
AMPLIFICATION_FACTOR = 30.0  # Amplify the audio by a factor of 2
OUTPUT_DIR = "/home/radxa/yamnet/Recorded_audio_Test"
SAMPLE_RATE = 16000  # Lower sample rate
CHANNELS = 1  # Mono
stop_recording = False

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def record_audio(filename):
    """Record audio using arecord with a fixed duration."""
    command = [
        'arecord', '-D', 'plughw:0', '-c', str(CHANNELS),
        '-r', str(SAMPLE_RATE), '-f', 'S16_LE', '-t', 'wav',
        '-d', str(RECORDING_DURATION),
        '-V', 'mono', '-v', filename
    ]
    result = subprocess.run(command)
    return result.returncode == 0  # Returns True if successful

def amplify_audio(audio_data):
    """Amplify the audio signal."""
    return np.clip(audio_data * AMPLIFICATION_FACTOR, -32768, 32767).astype(np.int16)

def apply_bandpass_filter(audio_data, lowcut=300.0, highcut=7500.0, sample_rate=SAMPLE_RATE):
    """Apply a bandpass filter to the audio data."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    # Ensure critical frequencies are within the range
    if not (0 < low < 1) or not (0 < high < 1):
        raise ValueError("Digital filter critical frequencies must be 0 < Wn < 1")
    b, a = butter(4, [low, high], btype='band')
    filtered_data = lfilter(b, a, audio_data)
    return filtered_data


def apply_noise_gate(audio_data, threshold=0.02):
    """Zero out values below a certain threshold to reduce background noise."""
    audio_data[np.abs(audio_data) < threshold] = 0
    return audio_data

def process_and_save_audio(input_filename, output_filename):
    """Amplify and filter the audio, then save it to a new file."""
    # Read the recorded audio
    sample_rate, audio_data = wavfile.read(input_filename)
    audio_data = audio_data.astype(np.float32)

    # Apply noise gate
    audio_data = apply_noise_gate(audio_data)

    # Apply amplification
    amplified_audio = amplify_audio(audio_data)

    # Apply bandpass filter
    filtered_audio = apply_bandpass_filter(amplified_audio)

    # Save the processed audio
    wavfile.write(output_filename, sample_rate, filtered_audio.astype(np.int16))

def stop_recording_listener():
    """Listen for Enter key to stop the recording loop."""
    global stop_recording
    input("Press Enter to stop recording...\n")
    stop_recording = True

def main():
    global stop_recording
    print("Recording... Press Enter to stop.")
    segment_count = 1

    # Start the stop recording listener in a separate thread
    listener_thread = threading.Thread(target=stop_recording_listener)
    listener_thread.start()

    try:
        while not stop_recording:
            # Define filenames
            raw_filename = os.path.join(OUTPUT_DIR, "temp_recording.wav")
            output_filename = os.path.join(OUTPUT_DIR, f"recording_{segment_count}_{int(time.time())}.wav")

            # Record audio and check if it was successful
            if not record_audio(raw_filename):
                print("Error: Unable to access recording device. Please check if it is available.")
                break

            # Process and save amplified and filtered audio
            process_and_save_audio(raw_filename, output_filename)
            print(f"Audio segment saved to {output_filename}")

            segment_count += 1

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stop_recording = True  # Ensure the listener thread ends
        listener_thread.join()  # Wait for the listener thread to complete
        print("Recording stopped.")

if __name__ == "__main__":
    main()
