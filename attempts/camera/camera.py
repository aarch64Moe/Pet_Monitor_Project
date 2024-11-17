import cv2
from flask import Response, stream_with_context

# New route to stream the camera feed
@app.route('/camera_stream')
def camera_stream():
    def generate():
        cap = cv2.VideoCapture(0)  # Change '0' if your USB camera has a different index
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_data = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        cap.release()

    # Pause classification while streaming
    global recording
    recording = False
    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame')
