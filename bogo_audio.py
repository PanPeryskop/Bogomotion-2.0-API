import requests
import os
import numpy as np
import pandas as pd
import audiofile
import torch
from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from pydub import AudioSegment
import torch.nn.functional as F


class BogoAudio:
    def __init__(self):
        self.model = AutoModelForAudioClassification.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.id2label = {
            "0": "angry",
            "1": "calm",
            "2": "disgust",
            "3": "fearful",
            "4": "happy",
            "5": "neutral",
            "6": "sad",
            "7": "surprised"
        }

    def save_audio_from_url(self, audio_url):
        if not audio_url:
            raise ValueError("No audio URL provided")

        file_extension = os.path.splitext(audio_url)[1]
        audio_name = "tmp_audio" + file_extension
        audio_path = os.path.join("audios/", audio_name)

        os.makedirs("audios/", exist_ok=True)

        try:
            audio = requests.get(audio_url)
            audio.raise_for_status()
        except requests.RequestException as e:
            raise Exception(f"Failed to download audio: {e}")

        with open(audio_path, 'wb') as f:
            f.write(audio.content)

        return audio_name

    def delete_audio(self, audio_name):
        audio = os.path.join("audios/", audio_name)
        if os.path.exists(audio):
            os.remove(audio)
        else:
            raise FileNotFoundError(f"Audio file {audio_name} not found")

    def process_audio(self, audio):
        audio = os.path.join("audios/", audio)

        if not os.path.exists(audio):
            raise FileNotFoundError(f"Audio file {audio} not found")

        return self.predict_emotion(audio)

    def predict_emotion(self, audio_file):
        sound = AudioSegment.from_file(audio_file)
        sound = sound.set_frame_rate(16000)
        sound_array = np.array(sound.get_array_of_samples())

        input = self.feature_extractor(
            raw_speech=sound_array,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt")

        result = self.model.forward(input.input_values.float())

        probabilities = F.softmax(result.logits, dim=-1)
        probabilities = probabilities[0].detach().numpy()

        interp = dict(zip(self.id2label.values(), list(round(float(i), 4) for i in probabilities)))
        return interp