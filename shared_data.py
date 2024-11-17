import threading


walking_count = 0
sitting_count = 0
recording_flag = False
last_recording_time = 0
movement = "Sitting/Standing Still"

# Shared variables
current_temperature = None
current_humidity = None
current_heart_rate = None
current_movement = None
current_classification = None
is_running = True


bus_number = 4

lock = threading.Lock()
