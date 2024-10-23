FROM python:3.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p model && cd model && \
    wget https://pjreddie.com/media/files/yolov3-tiny.weights && \
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg && \
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

COPY . .

CMD ["python", "app.py"]