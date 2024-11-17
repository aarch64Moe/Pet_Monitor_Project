from flask import Flask, jsonify, Response, stream_with_context

from flask_cors import CORS
import cv2
import threading
import time

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
terminate_flag = False  # Flag to indicate termination


@app.route('/camera_stream', methods=['GET'])
def camera_stream():
    global paused
    paused = True

    def generate():
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Error: Could not open the camera.")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            cap.release()
            paused = False
            print("Camera stream ended.")

    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    # Simple homepage with video stream
    return '''
    <html>
        <head>
            <title>Radxa Camera Stream</title>
        </head>
        <body>
            <h1>Camera Stream</h1>
            <img src="/camera_stream" style="width: 100%; height: auto;">
        </body>
    </html>
    '''


def input_listener():
    global terminate_flag
    input("Press Enter to terminate the application...\n")
    terminate_flag = True
    print("Termination signal received.")


if __name__ == "__main__":
    # Start the input listener in a separate thread
    threading.Thread(target=input_listener, daemon=True).start()

    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)
    print("Application terminated.")
