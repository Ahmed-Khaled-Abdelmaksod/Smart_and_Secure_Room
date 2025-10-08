from flask import Flask,request,jsonify,send_file,Response
import json
import numpy as np
import cv2
from faceRecognitionSystem import FaceRecognitionSystem
import io
app = Flask(__name__)
system = FaceRecognitionSystem(threshold=0.6)
unauth_photo = 0
@app.route("/",methods=['POST'])
def root_post_function(): 
    data = request.json
    print (data)
    return data

@app.route("/",methods=['GET'])
def root_get_function():
    return "Hello Auth System :)"

@app.route("/predict",methods=['POST'])
def security_check():
    if 'image' not in request.files:
        return jsonify({"error": "Couldn't find image"}) , 400
    file = request.files['image']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'cannot decode image'}), 400

    ### Enter to AI model
    # cv2.imshow("Preview", img)
    # cv2.waitKey(1)  # Wait until key press
    # cv2.destroyAllWindows()
    return jsonify({'Access': 'Auth'}) , 200


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
    if not authorized:
        global unauth_photo
        unauth_photo = frame
    
    return jsonify({"authorized": str(authorized),"name": str(name if authorized else None),"similarity": float(similarity)})


# ========== REGISTER FACE ==========
@app.route("/register", methods=["POST"])
def register_face():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    name = data.get("name")
    base64_image = data.get("image")

    if not name or not base64_image:
        return jsonify({"error": "Missing name or image"}), 400

    # Decode the base64 image
    import base64
    import numpy as np
    import cv2

    try:
        image_bytes = base64.b64decode(base64_image)
        npimg = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Invalid image data: {e}"}), 400

    faces = system.detect_faces(frame)
    if not faces:
        return jsonify({"error": "No face detected"}), 400

    x1, y1, x2, y2 = faces[0]
    face_img = system.align_face(frame, (x1, y1, x2, y2))
    system.add_authorized_face(name, face_img)

    return jsonify({"success": True, "message": f"Registered {name} successfully"})


CLEAR_AFTER_SEND = True

@app.route("/unauthorized", methods=["GET"])
def get_unauthorized_person():
    global unauth_photo

    # If no unauthorized photo stored yet
    if isinstance(unauth_photo, int) and unauth_photo == 0:
        return jsonify({"error": "No unauthorized image available"}), 404

    # Encode the OpenCV image to JPEG
    ret, jpg = cv2.imencode('.jpg', unauth_photo)
    if not ret:
        return jsonify({"error": "Failed to encode image"}), 500

    img_bytes = jpg.tobytes()
    bio = io.BytesIO(img_bytes)
    bio.seek(0)

    # Optionally clear the stored photo after serving
    if CLEAR_AFTER_SEND:
        unauth_photo = 0

    # send_file will set Content-Type: image/jpeg
    return send_file(bio, mimetype='image/jpeg', as_attachment=False,
                     download_name='unauthorized.jpg')

camera = None
is_streaming = False
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

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=7000)
