# flask_server.py
from flask import Flask, request, jsonify, Response, send_file
from flask_cors import CORS
import json
import numpy as np
import cv2
import os
import uuid
from datetime import datetime
import paho.mqtt.client as mqtt
import threading
import time
from faceRecognitionSystem import FaceRecognitionSystem

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter web
system = FaceRecognitionSystem(threshold=0.6)

# MQTT Configuration
MQTT_BROKER = '65e275a531cf4f5eb698af1ff09c51a7.s1.eu.hivemq.cloud'
MQTT_PORT = 8883
MQTT_USERNAME = 'hivemq.webclient.1758974031560'
MQTT_PASSWORD = 'n8!0I9&aNUF>Yw4zs<vM'

# MQTT Topics
TOPIC_ALERTS = 'room/alerts'
TOPIC_ACCESS = 'room/access'
TOPIC_SECURITY = 'room/security'

# Initialize MQTT client
mqtt_client = mqtt.Client(client_id=f'flask-server-{int(time.time())}')
mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqtt_client.tls_set()

# Storage for captured images
UPLOAD_FOLDER = 'captured_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Video streaming variables
camera = None
is_streaming = False

# Track pending access decisions
pending_access_decisions = {}

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe(TOPIC_ACCESS)
    client.subscribe(TOPIC_SECURITY)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        topic = msg.topic
        
        if topic == TOPIC_ACCESS:
            # Handle access decision from dashboard
            person_id = payload.get('person_id')
            allow = payload.get('allow')
            print(f"Access decision for {person_id}: {'ALLOWED' if allow else 'DENIED'}")
            
            # Store decision
            pending_access_decisions[person_id] = {
                'allow': allow,
                'timestamp': datetime.now()
            }
            
            # If allowed, add to authorized faces
            if allow and person_id in system.pending_approvals:
                data = system.pending_approvals[person_id]
                face_img = cv2.imread(data['image_path'])
                if face_img is not None:
                    # Get name from dashboard or use person_id
                    name = f"Dashboard_Approved_{person_id[:8]}"
                    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (112, 112))
                    system.add_authorized_face(name, face_resized)
                    
                    # Remove from pending
                    del system.pending_approvals[person_id]
                    system.save_pending_approvals()
                    
                    print(f"✓ Added {name} to authorized users via dashboard approval")
            
        elif topic == TOPIC_SECURITY:
            # Handle security mode changes
            enabled = payload.get('enabled')
            print(f"Security mode: {'ENABLED' if enabled else 'DISABLED'}")
            
    except Exception as e:
        print(f"Error handling MQTT message: {e}")

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

# Connect to MQTT broker
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
except Exception as e:
    print(f"Failed to connect to MQTT: {e}")

@app.route("/", methods=['GET'])
def root_get_function():
    return jsonify({"status": "Smart Room Security System Active"}), 200

@app.route("/verify", methods=["POST"])
def verify_face():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    faces = system.detect_faces(frame)
    if not faces:
        return jsonify({"authorized": False, "reason": "No face detected"}), 200

    x1, y1, x2, y2 = faces[0]
    face_img = system.align_face(frame, (x1, y1, x2, y2))
    name, similarity, authorized = system.recognize_face(face_img)
    
    # If unauthorized, save image and send MQTT alert
    if not authorized:
        # Generate unique ID
        person_id = str(uuid.uuid4())
        
        # Save images using face recognition system's method
        capture_id = system.capture_unauthorized_person(frame, face_img, similarity, name)
        
        # Use the capture_id as person_id for consistency
        person_id = capture_id
        
        # Get the saved image paths
        if person_id in system.pending_approvals:
            data = system.pending_approvals[person_id]
            
            # Copy face image to Flask's upload folder for serving
            face_filename = f"{person_id}_face.jpg"
            face_path_flask = os.path.join(UPLOAD_FOLDER, face_filename)
            cv2.imwrite(face_path_flask, cv2.imread(data['image_path']))
            
            # Send MQTT alert to dashboard
            alert_data = {
                "type": "unauthorized_person",
                "person_id": person_id,
                "image_url": f"/images/{face_filename}",
                "timestamp": datetime.now().isoformat(),
                "similarity": float(similarity) if similarity else 0.0,
                "best_match": name if name else "Unknown"
            }
            
            mqtt_client.publish(TOPIC_ALERTS, json.dumps(alert_data))
            print(f"Sent unauthorized person alert to dashboard: {person_id}")
    
    return jsonify({
        "authorized": bool(authorized),
        "name": str(name if authorized else None),
        "similarity": float(similarity) if similarity else 0.0
    })

@app.route("/register", methods=["POST"])
def register_face():
    name = request.form.get("name")
    if not name:
        return jsonify({"error": "Missing 'name' parameter"}), 400

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    faces = system.detect_faces(frame)
    if not faces:
        return jsonify({"error": "No face detected"}), 400

    x1, y1, x2, y2 = faces[0]
    face_img = system.align_face(frame, (x1, y1, x2, y2))
    system.add_authorized_face(name, face_img)

    return jsonify({"success": True, "message": f"Registered {name} successfully"})

# Serve captured images
@app.route('/images/<filename>')
def serve_image(filename):
    # Try both Flask's upload folder and face recognition system's folder
    flask_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(flask_path):
        return send_file(flask_path, mimetype='image/jpeg')
    
    # Try unauthorized captures folder
    system_path = os.path.join(system.unauthorized_dir, filename)
    if os.path.exists(system_path):
        return send_file(system_path, mimetype='image/jpeg')
    
    return jsonify({"error": "Image not found"}), 404

# List all images (for debugging)
@app.route('/images', methods=['GET'])
def list_images():
    files = []
    
    # List from Flask upload folder
    if os.path.exists(UPLOAD_FOLDER):
        files.extend([f"flask/{f}" for f in os.listdir(UPLOAD_FOLDER)])
    
    # List from system unauthorized folder
    if os.path.exists(system.unauthorized_dir):
        files.extend([f"system/{f}" for f in os.listdir(system.unauthorized_dir)])
    
    return jsonify({"images": files, "total": len(files)})

# Video streaming endpoints
def generate_frames():
    global camera
    while is_streaming:
        if camera is None or not camera.isOpened():
            break
            
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if is_streaming:
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return jsonify({"error": "Stream not started"}), 400

@app.route('/stream/start', methods=['POST'])
def start_stream():
    global camera, is_streaming
    if not is_streaming:
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            is_streaming = True
            return jsonify({"status": "streaming"}), 200
        else:
            return jsonify({"error": "Failed to open camera"}), 500
    return jsonify({"status": "already streaming"}), 200

@app.route('/stream/stop', methods=['POST'])
def stop_stream():
    global camera, is_streaming
    is_streaming = False
    if camera:
        camera.release()
        camera = None
    return jsonify({"status": "stopped"}), 200

# Continuing flask_server.py

@app.route('/test/unauthorized', methods=['GET'])
def test_unauthorized():
    """Manually trigger an unauthorized person alert"""
    
    # Create a test image with text
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (540, 380), (50, 50, 50), -1)
    cv2.putText(img, "TEST INTRUDER", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(img, datetime.now().strftime("%H:%M:%S"), (220, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    person_id = str(uuid.uuid4())
    
    # Save image
    filename = f"{person_id}_face.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    cv2.imwrite(filepath, img)
    
    # Send MQTT alert
    alert_data = {
        "type": "unauthorized_person",
        "person_id": person_id,
        "image_url": f"/images/{filename}",
        "timestamp": datetime.now().isoformat(),
        "similarity": 0.45,
        "best_match": "Test Person"
    }
    
    mqtt_client.publish(TOPIC_ALERTS, json.dumps(alert_data))
    
    return jsonify({
        "success": True,
        "message": "Test unauthorized alert sent",
        "person_id": person_id,
        "dashboard_url": f"Check dashboard for alert"
    })

# System status endpoint
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "face_recognition": {
            "authorized_faces": len(system.authorized_faces),
            "pending_approvals": len(system.pending_approvals),
            "threshold": system.threshold
        },
        "mqtt": {
            "connected": mqtt_client.is_connected()
        },
        "streaming": {
            "active": is_streaming
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7000, debug=True)