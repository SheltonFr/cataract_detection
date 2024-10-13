import pickle
import numpy as np
import cv2
from flask import Flask, request, jsonify
from utils import extract_features

with open('../assets/random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../assets/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)



app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found'}), 400

    file = request.files['file']

    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    features = extract_features(image)

    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)

    prediction_label = label_encoder.inverse_transform(prediction)
    return jsonify({'prediction': prediction_label[0]})

if __name__ == '__main__':
    app.run(debug=True)