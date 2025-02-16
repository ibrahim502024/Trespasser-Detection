from flask import Flask, render_template, request, Response, jsonify
import os
import cv2
from threading import Thread, Event
from ultralytics import YOLO
from playsound import playsound

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv8 model
model = YOLO("C:/Users/ibaib/Videos/final/runs/detect/train4/weights/best.pt")  # Update with your YOLOv8 model path

# Global variables
alarm_event = Event()
alarm_thread = None
webcam_streaming = False

def play_alarm():
    """Plays an alarm sound in a separate thread."""
    while not alarm_event.is_set():
        playsound("C:/Users/ibaib/Videos/test_data/alarm.mp3")  # Replace with your alarm sound path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    global alarm_thread, alarm_event

    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('index.html', error="No file uploaded.")

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    ext = os.path.splitext(file.filename)[1].lower()

    # Process image
    if ext in ['.jpg', '.jpeg', '.png']:
        results = model(filepath)
        labels = results[0].names
        detected_classes = [labels[int(cls)] for cls in results[0].boxes.cls]
        detected_confidences = results[0].boxes.conf.tolist()

        trespasser_detected = False
        for cls, conf in zip(detected_classes, detected_confidences):
            if cls == "Trespasser" and conf >= 0.7:
                trespasser_detected = True
                if not alarm_thread or not alarm_thread.is_alive():
                    alarm_event.clear()
                    alarm_thread = Thread(target=play_alarm)
                    alarm_thread.start()

        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'output_{file.filename}')
        annotated_image = results[0].plot()
        cv2.imwrite(output_path, annotated_image)

        return render_template(
            'index.html',
            output_image=output_path,
            trespasser=trespasser_detected
        )

    # Process video (stream frame by frame)
    elif ext in ['.mp4', '.avi', '.mov']:
        return Response(gen_video(filepath), mimetype='multipart/x-mixed-replace; boundary=frame')

    else:
        return render_template('index.html', error="Unsupported file format.")

def gen_video(filepath):
    global alarm_thread, alarm_event

    cap = cv2.VideoCapture(filepath)
    trespasser_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the current frame
        results = model(frame)
        labels = results[0].names

        for box in results[0].boxes:
            cls = int(box.cls)
            confidence = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            if labels[cls] == "Trespasser" and confidence >= 0.7:
                trespasser_detected = True
                if not alarm_thread or not alarm_thread.is_alive():
                    alarm_event.clear()
                    alarm_thread = Thread(target=play_alarm)
                    alarm_thread.start()

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{labels[cls]}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/start_webcam')
def start_webcam():
    global webcam_streaming
    webcam_streaming = True
    return Response(gen_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_streaming
    webcam_streaming = False
    return jsonify({"status": "Webcam stopped."})

def gen_webcam():
    global webcam_streaming, alarm_thread, alarm_event

    cap = cv2.VideoCapture(0)

    while webcam_streaming:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection on the current frame
        results = model(frame)
        labels = results[0].names

        for box in results[0].boxes:
            cls = int(box.cls)
            confidence = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates

            if labels[cls] == "Trespasser" and confidence >= 0.7:
                if not alarm_thread or not alarm_thread.is_alive():
                    alarm_event.clear()
                    alarm_thread = Thread(target=play_alarm)
                    alarm_thread.start()

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{labels[cls]}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm():
    global alarm_event
    alarm_event.set()  # Stop the alarm
    return jsonify({"status": "Alarm stopped."})

@app.route('/reset', methods=['POST'])
def reset():
    global alarm_event
    alarm_event.set()  # Stop the alarm

    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    return jsonify({"status": "Reset complete."})

if __name__ == '__main__':
    app.run(debug=True)
