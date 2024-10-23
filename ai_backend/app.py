import os
import cv2
import numpy as np
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

net = cv2.dnn.readNet("model/yolov3-tiny.weights", "model/yolov3-tiny.cfg")
with open("model/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'})
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)
    
    img = cv2.imread(filepath)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)  

    results = []
    if isinstance(indexes, np.ndarray):
        indexes = indexes.flatten()
    
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            results.append({
                'label': label,
                'confidence': round(float(confidence), 3),
                'box': {'x': x, 'y': y, 'w': w, 'h': h}
            })

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {confidence:.2f}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output_img_path = os.path.join('output', 'images', filename)
    cv2.imwrite(output_img_path, img)

    output_json_path = os.path.join('output', 'json', f'{os.path.splitext(filename)[0]}.json')
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    return jsonify(results)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('output/images', exist_ok=True)
    os.makedirs('output/json', exist_ok=True)
    app.run(host='0.0.0.0', port=5000)