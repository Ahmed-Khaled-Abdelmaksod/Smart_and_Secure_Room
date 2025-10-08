import serial
import json
import time
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO

# ---------------------------
# Arduino Serial Setup
# ---------------------------
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)

# ---------------------------
# MQTT Setup
# ---------------------------
BROKER = '65e275a531cf4f5eb698af1ff09c51a7.s1.eu.hivemq.cloud'
PORT = 8883
USER = 'hivemq.webclient.1758974031560'
PASSWORD = 'n8!0I9&aNUF>Yw4zs<vM'

data_topic = 'room/monitor/data'
door_topic = 'room/door/open'
verify_topic = 'room/camera/verify'
light_topic = 'room/light'
client = mqtt.Client()
client.username_pw_set(USER, PASSWORD)
client.tls_set()

# ---------------------------
# Servo Setup
# ---------------------------
SERVO_PIN = 18
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(SERVO_PIN, GPIO.OUT)
pwm = GPIO.PWM(SERVO_PIN, 50)  # 50Hz
pwm.start(0)


# -----------------------------
# Light Setup
# -----------------------------
LIGHT_PIN = 17
GPIO.setup(LIGHT_PIN, GPIO.OUT)
GPIO.output(LIGHT_PIN, GPIO.LOW)

def light_on():
    GPIO.output(LIGHT_PIN, GPIO.HIGH)
def light_off():
    GPIO.output(LIGHT_PIN, GPIO.LOW)

def set_angle(angle):
    duty = 2.5 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    pwm.ChangeDutyCycle(0)

def open_door():
    print("🚪 Opening door...")
    set_angle(0)
    time.sleep(2)  # door open duration
    print("🔒 Closing door...")
    set_angle(95)

# ---------------------------
# MQTT Handlers
# ---------------------------
def on_connect(client, userdata, flags, rc, properties=None):
    print("✅ Connected to MQTT Broker")
    client.subscribe(door_topic)

def on_message(client, userdata, msg):
    print(f"📩 Message received: {msg.topic} | {msg.payload.decode()}")
    if msg.topic == door_topic:
       open_door()
    elif msg.topic == light_topic:
       print(msg.payload.decode())
       data = json.loads(msg.payload.decode())
       if data['light'] == True:
          light_on()
          print("Light On")
       elif data['light'] == False:
          light_off()
          print("Light Off")
client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER, PORT)
client.subscribe(light_topic)
client.loop_start()

# ---------------------------
# Main Loop
# ---------------------------
try:
    while True:
        # --- Read Arduino Data ---
        line = arduino.readline().decode('utf-8').strip()
        if line:
            try:
                data = json.loads(line)
                data_without_distance = data
                #del data_without_distance['distance']
                print(f"🌡️ {data['temperature']} °C | 💧 {data['humidity']} % | 👀 Motion: {data['motion']} | 💨 Smoke: {data['smoke']}  , Distance: {data['distance']}")
                client.publish(data_topic, json.dumps(data_without_distance), qos=1)
                if data['distance'] < 15:
                   client.publish(verify_topic,json.dumps({"Person":"True"}))
            except json.JSONDecodeError:
                print(f"⚠️ Invalid JSON: {line}")

        time.sleep(1)

except KeyboardInterrupt:
    print("🛑 Exiting...")
finally:
    pwm.stop()
    GPIO.cleanup()
    client.loop_stop()
    client.disconnect()

