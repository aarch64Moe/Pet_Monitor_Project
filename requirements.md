
# Requirements for Radxa Zero 3W Pet Monitoring System

This document lists all the dependencies and system configurations needed to set up the environment for the AI-powered pet monitoring system.

---

## System-Level Packages

Use the following commands to install the required system libraries:

```bash
sudo apt update
sudo apt install -y libsndfile1 libportaudio2 libatlas-base-dev python3-venv i2c-tools alsa-utils libopencv-dev python3-opencv
```

---

## Python Environment Setup

### Step 1: Create a Virtual Environment
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Step 2: Install Python Libraries

Install the following Python dependencies:
```bash
pip install "numpy<2" scikit-learn joblib scipy smbus2 sounddevice soundfile opencv-python pyserial Flask flask-cors
```

---

## Additional Notes

### Custom Modules

The project references custom modules such as:
- `shared_data`
- `filters`
- `mpu`
- `calibration`
- `Temperature_sensor`
- `heart`
- `DeviceNum`

Ensure these files are part of your project directory.

### Sensor-Specific Libraries

For sensor integration:
- **I2C Tools**: Required for sensors like DS18B20 and MPU6050.
  ```bash
  sudo apt install i2c-tools
  ```
- **Audio Drivers**: Ensure ALSA is configured for `sounddevice` to work.
  ```bash
  sudo apt install alsa-utils
  ```

---

## Install All Python Dependencies with `requirements.txt`

You can also save the following list to a `requirements.txt` file:

```plaintext
Flask==2.3.2
flask-cors==3.0.10
numpy<2
scikit-learn==1.3.1
joblib
scipy
smbus2
sounddevice==0.4.6
soundfile
opencv-python==4.8.1.78
pyserial
tflite-runtime==2.10.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

