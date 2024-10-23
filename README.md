# Object Detection Microservice using YOLOv3

This project implements a microservice for object detection using the ultralytics/yolov3 implementation. It consists of a UI backend service for handling user uploads and an AI backend service for performing object detection.

## Prerequisites

- Docker
- Docker Compose
- Git

## Setup

1. Clone this repository:
```bash
git clone https://github.com/your-username/object-detection-microservice.git
cd object_detection_microservice
```

2. Clone the YOLOv3 repository and download weights:
```bash
git clone https://github.com/ultralytics/yolov3.git ai_backend/yolov3
wget -P ai_backend/yolov3/ https://github.com/ultralytics/yolov3/releases/download/v9.0/yolov3.pt
```

3. Build and run the Docker containers:
```bash
docker-compose up --build
```

## Usage

1. Open a web browser and navigate to `http://localhost:8080`
2. Upload an image using the provided form
3. The system will process the image using YOLOv3 and return the detected objects
4. Results can be found in:
   - `output/images/`: Processed images with detection boxes
   - `output/json/`: JSON files containing detection results

## Project Structure

```
object_detection_microservice/
├── ui_backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── templates/
│       └── upload.html
├── ai_backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app.py
│   └── yolov3/  # Ultralytics YOLOv3 repository
├── output/
│   ├── images/
│   └── json/
└── docker-compose.yml
```

## Detection Results

The system returns results in the following format:
```json
[
  {
    "label": "person",
    "confidence": 0.95,
    "box": {
      "x1": 100,
      "y1": 200,
      "x2": 300,
      "y2": 400
    }
  }
]
```

## References

- YOLOv3: https://github.com/ultralytics/yolov3
- Flask: https://flask.palletsprojects.com/