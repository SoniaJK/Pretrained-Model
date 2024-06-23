
from flask import Flask, request, jsonify
from model.gesture_model import load_model
from src.landmarks_extraction import mediapipe_detection, draw, extract_coordinates
from src.config import SEQ_LEN, THRESH_HOLD
import numpy as np
import cv2
import mediapipe as mp

app = Flask(__name__)
model = load_model()

mp_holistic = mp.solutions.holistic

@app.route('/predict', methods=['POST'])
def predict():
    sequence_data = []
    data = request.json
    frame_data = np.frombuffer(bytearray(data['frame']), dtype=np.uint8)
    frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image, results = mediapipe_detection(frame, holistic)
        draw(image, results)
        
        try:
            landmarks = extract_coordinates(results)
        except:
            landmarks = np.zeros((468 + 21 + 33 + 21, 3))
        sequence_data.append(landmarks)
        
        if len(sequence_data) % SEQ_LEN == 0:
            prediction = model(np.array(sequence_data, dtype=np.float32))["outputs"]
            if np.max(prediction.numpy(), axis=-1) > THRESH_HOLD:
                sign = np.argmax(prediction.numpy(), axis=-1)
                sequence_data = []
                return jsonify({'sign': sign})
        
    return jsonify({'sign': None})

if __name__ == '__main__':
    app.run(debug=True)
