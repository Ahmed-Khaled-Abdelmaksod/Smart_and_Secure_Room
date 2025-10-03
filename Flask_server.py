from flask import Flask,request,jsonify
import json
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/",methods=['POST'])
def root_function():
    data = request.json
    print (data)
    return data

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

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=7000)