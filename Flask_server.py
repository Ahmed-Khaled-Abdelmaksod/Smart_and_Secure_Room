from flask import Flask,request,jsonify
import json
import numpy as np
import cv2
from faceRecognitionSystem import FaceRecognitionSystem

app = Flask(__name__)
system = FaceRecognitionSystem(threshold=0.6)

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
    
    return jsonify({"authorized": str(authorized),"name": str(name if authorized else None),"similarity": float(similarity)})


# ========== REGISTER FACE ==========
@app.route("/register", methods=["POST"])
def register_face():
    """
    POST /register
    Form-data:
      - name: person's name
      - image: uploaded image file
    """
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


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=7000)