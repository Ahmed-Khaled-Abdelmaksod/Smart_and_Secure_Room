import cv2
import paho.mqtt.client as mqtt 
import requests
import json


BROKER = '65e275a531cf4f5eb698af1ff09c51a7.s1.eu.hivemq.cloud'
PORT = 8883
USER = 'hivemq.webclient.1758974031560'
PASSWORD = 'n8!0I9&aNUF>Yw4zs<vM'
FLASK_SERVER_VERIFY_URL = 'http://localhost:7000/verify'
FLASK_SERVER_REGISTER_URL = 'http://localhost:7000/register'
camera_register_topic = 'room/camera/register'
camera_verify_topic = 'room/camera/verify'
room_door_topic = "room/door/open"
unauth_signal_topic = 'room/gate/unauth'
REGISTER_CMD = 0
VERIFY_CMD = 1
client = mqtt.Client()
client.username_pw_set(USER,PASSWORD)
client.tls_set()


def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Can't open camera")
    else:
        ret , frame = cap.read()
        return (ret,frame)
        # if ret:
        #     cv2.imshow("Preview", frame)
        #     cv2.waitKey(0)  # Wait until key press
        #     cv2.destroyAllWindows()
def send_to_server(frame,cmd,name=None):
    ret,jpg = cv2.imencode('.jpg',frame)
    img_bytes = jpg.tobytes()
    files = {
        "image": ('capture.jpg', img_bytes, 'image/jpeg')
    }
    data = {"name":'Ahmed'}
    if cmd == REGISTER_CMD:
        resp = requests.post(FLASK_SERVER_REGISTER_URL, files=files, data=data,timeout=15)
        print("Res:",resp.status_code," : ",resp.text)
    elif cmd == VERIFY_CMD:
        resp = requests.post(FLASK_SERVER_VERIFY_URL, files=files, timeout=15)
        resp_json = resp.json()
        if resp_json.get('authorized') == 'True':
            client.publish(room_door_topic, payload="Authorized")
        elif resp_json.get('authorized') == 'False':
            client.publish(unauth_signal_topic, payload=json.dumps({"unauth": True}))
            
        # if resp['authorized']:
        #     client.publish(room_door_topic,payload=str("Authorized"))
        print("Res:",resp.status_code," : ",resp.text)


def on_message(client, userdata, msg):
    print("message:",msg.payload)
    ret,frame = capture_image()
    if msg.topic == camera_register_topic :
        if ret:
            send_to_server(frame,REGISTER_CMD)
    elif msg.topic == camera_verify_topic:
        if ret:
            send_to_server(frame,VERIFY_CMD)

def on_connect(client, userdata, flags, rc, properties=None):
    print("connected :)")


client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER,PORT)


client.subscribe(camera_register_topic,qos=1)
client.subscribe(camera_verify_topic,qos=1)




client.loop_forever()
