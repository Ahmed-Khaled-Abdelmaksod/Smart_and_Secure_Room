import cv2
import paho.mqtt.client as mqtt 
import requests
import json
import time
from datetime import datetime

# MQTT Configuration
BROKER = '65e275a531cf4f5eb698af1ff09c51a7.s1.eu.hivemq.cloud'
PORT = 8883
USER = 'hivemq.webclient.1758974031560'
PASSWORD = 'n8!0I9&aNUF>Yw4zs<vM'

# Flask server URLs
FLASK_SERVER_VERIFY_URL = 'http://localhost:7000/verify'
FLASK_SERVER_REGISTER_URL = 'http://localhost:7000/register'

# MQTT Topics
camera_register_topic = 'room/camera/register'
camera_verify_topic = 'room/camera/verify'
access_topic = 'room/access'
people_count_topic = 'room/sensors/people'

# Commands
REGISTER_CMD = 0
VERIFY_CMD = 1

# Initialize MQTT client
client = mqtt.Client(client_id=f'camera-client-{int(time.time())}')
client.username_pw_set(USER, PASSWORD)
client.tls_set()

# People counter (simple implementation)
people_count = 0

def capture_image():
    """Capture a single frame from camera"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open camera")
        return (False, None)
    else:
        ret, frame = cap.read()
        cap.release()
        return (ret, frame)

def send_to_server(frame, cmd, name=None):
    """Send captured frame to Flask server"""
    ret, jpg = cv2.imencode('.jpg', frame)
    img_bytes = jpg.tobytes()
    files = {
        "image": ('capture.jpg', img_bytes, 'image/jpeg')
    }
    
    try:
        if cmd == REGISTER_CMD:
            data = {"name": name or 'Unknown'}
            resp = requests.post(FLASK_SERVER_REGISTER_URL, files=files, data=data, timeout=15)
            print(f"Register response: {resp.status_code} : {resp.text}")
        elif cmd == VERIFY_CMD:
            resp = requests.post(FLASK_SERVER_VERIFY_URL, files=files, timeout=15)
            print(f"Verify response: {resp.status_code} : {resp.text}")
            
            # Parse response and update people count
            result = resp.json()
            if result.get('authorized'):
                update_people_count(1)  # Person entered
            
    except Exception as e:
        print(f"Error sending to server: {e}")

def update_people_count(change):
    """Update and publish people count"""
    global people_count
    people_count += change
    people_count = max(0, people_count)  # Don't go negative
    
    # Publish to MQTT
    data = {
        "people_count": people_count,
        "timestamp": datetime.now().isoformat()
    }
    client.publish(people_count_topic, json.dumps(data))
    print(f"Updated people count: {people_count}")

def on_message(client, userdata, msg):
    """Handle MQTT messages"""
    print(f"Message on {msg.topic}: {msg.payload.decode()}")
    
    try:
        if msg.topic == camera_register_topic:
            # Register new face
            payload = json.loads(msg.payload.decode())
            name = payload.get('name', 'Unknown')
            ret, frame = capture_image()
            if ret:
                send_to_server(frame, REGISTER_CMD, name)
                
        elif msg.topic == camera_verify_topic:
            # Verify face
            ret, frame = capture_image()
            if ret:
                send_to_server(frame, VERIFY_CMD)
                
        elif msg.topic == access_topic:
            # Handle access decision from dashboard
            payload = json.loads(msg.payload.decode())
            person_id = payload.get('person_id')
            allow = payload.get('allow')
            
            if allow:
                print(f"Opening door for person {person_id}")
                # Add your door control logic here
                update_people_count(1)  # Person entered
            else:
                print(f"Access denied for person {person_id}")
                
    except Exception as e:
        print(f"Error handling message: {e}")

def on_connect(client, userdata, flags, rc):
    """Handle MQTT connection"""
    print(f"Connected to MQTT broker with code: {rc}")
    # Subscribe to topics
    client.subscribe(camera_register_topic, qos=1)
    client.subscribe(camera_verify_topic, qos=1)
    client.subscribe(access_topic, qos=1)
    print("Subscribed to topics")

# Set callbacks
client.on_connect = on_connect
client.on_message = on_message

# Connect to MQTT broker
print("Connecting to MQTT broker...")
client.connect(BROKER, PORT, keepalive=60)

# Start periodic people detection (optional)
def periodic_people_detection():
    """Periodically check for people and update count"""
    while True:
        try:
            # Simulate people detection (you can integrate with your actual detection)
            # This is just a placeholder - integrate with your actual people counting logic
            time.sleep(30)  # Check every 30 seconds
        except Exception as e:
            print(f"Error in people detection: {e}")

# Start MQTT loop
print("Starting camera monitoring system...")
client.loop_forever()