import cv2

def start_camera_stream():
    try:
        # Open the camera (default is device index 0)
        camera = cv2.VideoCapture(0)

        # Set a lower resolution to reduce resource usage
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)  # Lower frame rate if needed

        if not camera.isOpened():
            print("Error: Could not open the camera.")
            return

        print("Camera stream started. Press 'q' to quit.")

        # Stream loop
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Error: Unable to read from the camera.")
                break

            # Display the video stream
            cv2.imshow("Camera Stream", frame)

            # Detect key press
            key = cv2.waitKey(10) & 0xFF
            #print(f"Key pressed: {key}")  # Debugging
            if key == ord('q'):
                print("Quitting stream...")
                break


    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release resources
        camera.release()
        cv2.destroyAllWindows()
        print("Camera stream stopped.")

if __name__ == "__main__":
    start_camera_stream()
