import cv2
import paho.mqtt.client as mqtt
import requests

BROKER = '65e275a531cf4f5eb698af1ff09c51a7.s1.eu.hivemq.cloud'
PORT = 8883
USER = 'hivemq.webclient.1758974031560'
PASSWORD = 'n8!0I9&aNUF>Yw4zs<vM'
FLASK_SERVER_URL = 'http://localhost:7000/predict'

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
def send_to_server(frame):
    ret,jpg = cv2.imencode('.jpg',frame)
    img_bytes = jpg.tobytes()
    files = {
        "image": ('capture.jpg', img_bytes, 'image/jpeg')
    }
    resp = requests.post(FLASK_SERVER_URL, files=files, timeout=15)
    print("Res:",resp.status_code," : ",resp.text)


def on_message(client, userdata, msg):
    print("message:",userdata)
    ret,frame = capture_image()
    if ret:
        send_to_server(frame)

def on_connect(client, userdata, flags, rc, properties=None):
    print("connected :)")


client.on_connect = on_connect
client.on_message = on_message
client.connect(BROKER,PORT)
camera_topic = 'room/camera/capture'

client.subscribe(camera_topic,qos=1)




client.loop_forever()