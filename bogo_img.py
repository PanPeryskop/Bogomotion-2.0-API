from PIL import Image
from io import BytesIO
import os
import math
import os
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import requests


class BogoImage:
    def __init__(self):
        self.class_names = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }

        self.model = YOLO("models/bogo300.pt")

    def save_img_from_url(self, img_url, img_name):
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
        img.save(img_name)

    def get_emotion_from_img(self, img_name):
        img = cv2.imread(img_name)
        results = self.model(img)

        infos = []
        highest_confidence_box = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = math.ceil((box.conf[0] * 100)) / 100
                class_id = int(box.cls[0])
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    infos.append({'emotion': class_name, 'confidence': confidence, 'box': box})

        if infos:
            highest_confidence_info = max(infos, key=lambda x: x['confidence'])
            highest_confidence_box = highest_confidence_info['box']
            x1, y1, x2, y2 = highest_confidence_box.xyxy[0]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
            print(f"Box: {x1, y1, x2, y2}")
            output = {'error': "false", 'emotion': highest_confidence_info['emotion'], 'w': x1, 'x': y1, 'y': x2, 'z': y2}
            return output

        else:
            print("No recognized emotions found.")
            return {'error': 'true', 'emotion': None, 'w': None, 'x': None, 'y': None, 'z': None}

    def delete_img(self, img_name):
        os.remove(img_name)




