from flask import Flask, request, jsonify
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
import requests

from bogo_audio import BogoAudio
from bogo_img import BogoImage
from noise import BogoNoise
from bogo_qual import BogoQualityChecker
from bogo_llm import BogoLlm

from PIL import Image
from io import BytesIO

import json

import math
import os
import cv2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
api = Api(app)

parser = reqparse.RequestParser()

noise = BogoNoise()

bogollm = BogoLlm()

class Emotion(Resource):
    def post(self):
        data = request.get_json()
        print(data)
        key = data.get('key')

        if key != 'img_em' and key != 'audio_em' and key != 'noise' and key != 'img_qual' and key != 'llm':
            return {'message': 'Invalid key provided'}, 400

        elif key == 'audio_em':
            audio_url = data.get("audio_url")
            if audio_url is None:
                return {'message': 'No audio URL provided'}, 400

            audiob = BogoAudio()
            audio = audiob.save_audio_from_url(audio_url=audio_url)
            info = audiob.process_audio(audio=audio)
            audiob.delete_audio(audio_name=audio)
            info = {'response': info}, 200
            return info
        elif key == 'noise':
            audio_url = data.get("audio_url")
            if audio_url is None:
                return {'message': 'No audio URL provided'}, 400
            noise.save_audio_from_url(audio_url=audio_url)
            noisy =  noise.is_noisy()
            if noisy is not None:
                if noisy:
                    return {'response': 'noisy'}, 200
                else:
                    return {'response': 'not-noisy'}, 200
            else:
                return {'response': 'error'}, 400

        elif key == 'img_qual':

            img_url = data.get('img_url')
            if img_url is None:
                return {'message': 'No image provided'}, 400
            img = 'tmp'
            ext = os.path.splitext(img_url)[1]
            img_name = img + ext
            response = requests.get(img_url)
            img = Image.open(BytesIO(response.content))
            img.save(img_name)
            if not os.path.exists(img_name):
                return {'message': 'Image not found'}, 400
            qual = BogoQualityChecker(img_name)
            img_info = qual.classify_quality()
            output = {
                'quality_score': img_info['quality_score'],
                'tests_passed': img_info['tests_passed'],
                'total_tests': img_info['total_tests'],
                'resolution_passed': img_info['resolution_passed']
            }
            return {'response': output}, 200

        elif key == 'llm':
            prompt = data.get('prompt')
            if prompt is None:
                return {'message': 'No prompt provided'}, 400
            output = bogollm.generate(prompt)
            return {'response': output}, 200

        else:
            bimg = BogoImage()
            img_url = data.get('img_url')

            if img_url is None:
                return {'message': 'No image provided'}, 400

            name = 'tmp'
            ext = os.path.splitext(img_url)[1]
            img_name = name + ext
            print(img_name)

            bimg.save_img_from_url(img_url, img_name)

            while not os.path.exists(img_name):
                pass

            emotion = bimg.get_emotion_from_img(img_name)
            bimg.delete_img(img_name)

            output = {
                'response': emotion
            }

            print(output)

            return output, 200


api.add_resource(Emotion, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5048)