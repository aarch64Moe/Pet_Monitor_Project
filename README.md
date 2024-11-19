# AI-Powered Pet Monitoring System

This project is a hardware-software integrated system designed to monitor pet health and behavior in real-time. The system leverages sensors, a camera, and machine learning models to analyze and classify pet activities and provide health insights.

---

## 🛠️ Hardware Architecture / Backend Mobile App System Architecture 

                                                  [ LiPo Battery ]
                                                         |
                                            [ Power Management Circuit ]
                                                         |
    ----------------------------------------------------------------------------------------------------------------
    |                                                                                                              |
    |                  [ Microcontroller (Quad-core ARM Cortex-A55 @ 1.6GHz + Mali-G52 GPU) ]                      |
    |                                                                                                              |
    |         [i2c Bus 4]                [uart2-m0]    [MIPI CSI]   [I2S BUS 3]            [Runs Locally On eMMC]  |
    ---------------|-------------------------|-------------|-------------|----------------------------|-------------
        -----------------------              |             |             |             |              |
        |          |          |              |             |             |             |              | 
    [ Heart ]   [ Temp ]   [ Movment ]  [GPS Module]    [Camera]   [Microphone]     [Comm]         [Yamnet]
    [MAX30102]  [DS18B20]  [ MPU6050 ]  [VK28U7G5LF]    [Module]     [INMP441]    [Wi-Fi/Blu]       [MLM]
                                                                                  [TTP/HTTPS]
                                                                                       |
                                                                                       |
                                                                                       |                                         
                                                                     ------------------|----------------------
                                                                     |                                       |
                                                                     |         [Mobile Application]          |
                                                                     |             [Flask API]               |
                                                                     |                                       |
                                                                     -----------------------------------------
                                                                            |                     |                                                  
                                                                      [Cloud Server]     [Push Notifications]
                                                                        [AWS EC2]               [FCM]
  
---

## 🧠 Software Overview

The system combines real-time data collection with AI-powered classification models to detect pet behavior and health parameters.

### Key Features:
- **Machine Learning**: YAMNet with a custom classification layer for detecting pet vocalizations.
- **Real-Time Monitoring**:
  - Heart rate detection using MAX30102.
  - Temperature and humidity tracking using DS18B20 sensor.
  - Movement detection using MPU6050 accelerometer and gyroscope.
- **Audio Analysis**:
  - Filters audio input using a bandpass filter (300Hz - 5000Hz).
  - Classifies vocalizations and behaviors using TensorFlow Lite models.
- **Camera Integration**:
  - Live video stream from ArduCam or Pi Camera.
  - Supports video-based behavior analysis.
- **GPS Tracking**:
  - Monitors pet location using the VK28U7G5LF GPS module.

---

## 🖥️ Software Components

### Sensors Integration
- **Heart Rate Sensor (MAX30102)**: Measures heart rate.
- **Temperature Sensor (DS18B20)**: Tracks ambient and body temperature.
- **GPS Module (VK28U7G5LF)**: Provides real-time location tracking.
- **MPU6050**: Detects movement and orientation.

### Machine Learning Models
- **YAMNet**: A pretrained audio classification model fine-tuned for pet vocalizations.
- **Custom Classifier**: Classifies behaviors like:
  - Angry
  - Fighting
  - Happy
  - Hunting
  - Purring

### Backend System
- Flask API to serve real-time data to mobile apps.
- Routes:
  - `/camera_stream`: Streams live video.
  - `/gps`   : Provides real-time GPS location data.
  - `/status`: Provides real-time sensor and classification data.
  - `/reboot`: Reboots the system.

---

## 🚀 Getting Started

### Prerequisites
1. Python 3.8 or higher
2. TensorFlow Lite runtime
3. Flask and other dependencies (`requirements.txt` provided)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aarch64Moe/Pet_Monitor_Project.git
   cd Pet_Monitor_Project

