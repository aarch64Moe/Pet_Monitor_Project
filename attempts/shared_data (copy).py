import threading


walking_count = 0
sitting_count = 0
recording_flag = False
last_recording_time = 0
movement = "Sitting/Standing Still"
bus_number = 4

lock = threading.Lock()
