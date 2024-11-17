import cv2

def open_camera_stream():
    """Opens the camera stream and displays it."""
    cap = cv2.VideoCapture(0)  # Open the default camera (USB camera should be index 0 or try 1 if needed)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print("Press 'q' to exit the camera stream.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Camera Stream", frame)

        # Press 'q' to close the stream
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def capture_image(filename="captured_image.jpg"):
    """Captures a single image from the camera and saves it."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    ret, frame = cap.read()
    if ret:
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")
    else:
        print("Failed to capture image.")

    cap.release()

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Open Camera Stream")
    print("2. Capture Image")

    choice = input("Enter choice (1/2): ")

    if choice == "1":
        open_camera_stream()
    elif choice == "2":
        filename = input("Enter filename for the image (e.g., 'image.jpg'): ")
        capture_image(filename)
    else:
        print("Invalid choice. Please enter 1 or 2.")
