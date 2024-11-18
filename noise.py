from pydub import AudioSegment
from pydub.silence import detect_silence
import os
import requests


class BogoNoise:
    def __init__(self):
        self.ext = None

    def is_noisy(self):
        path = "noise_test/tmp_audio"
        audio_file = path + self.ext
        if os.path.exists(audio_file):
            audio = AudioSegment.from_file(audio_file)
            silence = detect_silence(audio, min_silence_len=1000, silence_thresh=-40)
            if len(silence) > 0:
                return False
            else:
                return True
        else:
            return None

    def save_audio_from_url(self, audio_url):
        if not audio_url:
            raise ValueError("No audio URL provided")

        file_extension = os.path.splitext(audio_url)[1]
        self.ext = file_extension
        audio_name = "tmp_audio" + file_extension
        audio_path = os.path.join("noise_test/", audio_name)

        os.makedirs("noise_test/", exist_ok=True)

        try:
            audio = requests.get(audio_url)
            audio.raise_for_status()
        except requests.RequestException as e:
            raise Exception(f"Failed to download audio: {e}")

        with open(audio_path, 'wb') as f:
            f.write(audio.content)

        return audio_name

    def delete_audio(self, audio_name):
        audio = os.path.join("noise_test/", audio_name)
        if os.path.exists(audio):
            os.remove(audio)
        else:
            raise FileNotFoundError(f"Audio file {audio_name} not found")