# BogoMotion 2.0 API

This project is an API for the [BogoMotion2](https://github.com/azizko1337/bogomotion2). It provides various functionalities including image quality checking, emotion detection from images and audio, auto labeling, noise detection, and chat api based on llama2.

**This project won 2nd place in the 2nd edition of the HackEmotion 24h hackathon organized by the University of Silesia.**


## Modules

### 1. Image Quality Checker
This module checks the quality of an image based on several metrics such as resolution, blurriness, brightness, saturation, noise, and exposure.

### 2. Emotion Detection from Images
This module uses a YOLO model to detect emotions from images. The model has an accuracy of 81%. Below are some performance metrics and visualizations:

| Image | Description |
|-------|-------------|
| <img src="https://raw.githubusercontent.com/PanPeryskop/Bogomotion-2.0-API/refs/heads/main/runs/detect/train2/val_batch0_pred.jpg" alt="Predictions" width="300"/> | *Predictions* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/Bogomotion-2.0-API/refs/heads/main/runs/detect/train2/val_batch0_labels.jpg" alt="Labels" width="300"/> | *Labels* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/Bogomotion-2.0-API/refs/heads/main/runs/detect/train2/labels.jpg" alt="Labels" width="300"/> | *Labels* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/Bogomotion-2.0-API/refs/heads/main/runs/detect/train2/confusion_matrix_normalized.png" alt="Confusion Matrix" width="300"/> | *Confusion Matrix* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/Bogomotion-2.0-API/refs/heads/main/runs/detect/train2/R_curve.png" alt="R Curve" width="300"/> | *R Curve* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/Bogomotion-2.0-API/refs/heads/main/runs/detect/train2/P_curve.png" alt="P Curve" width="300"/> | *P Curve* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/Bogomotion-2.0-API/refs/heads/main/runs/detect/train2/PR_curve.png" alt="PR Curve" width="300"/> | *PR Curve* |
| <img src="https://raw.githubusercontent.com/PanPeryskop/Bogomotion-2.0-API/refs/heads/main/runs/detect/train2/F1_curve.png" alt="F1 Curve" width="300"/> | *F1 Curve* |

### 3. Emotion Detection from Audio
This module uses a pre-trained Wav2Vec2 model to detect emotions from audio files.

### 4. Noise Detection
This module detects whether an audio file is noisy or not.

### 5. Language Model Generation
This module uses a Llama model to generate text based on a given prompt. The model can be downloaded from [Hugging Face](https://huggingface.co/TheBloke/Llama-2-7B-GGUF).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/PanPeryskop/Bogomotion-2.0-API
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Install PyTorch from [PyTorch's official site](https://pytorch.org/get-started/locally/).

## Usage

1. **Run the API**:
    ```bash
    python bogo_api.py
    ```

2. **Send a POST request to the API**:
    - For image emotion detection:
        ```json
        {
            "key": "img_em",
            "img_url": "URL_TO_IMAGE"
        }
        ```
    - For audio emotion detection:
        ```json
        {
            "key": "audio_em",
            "audio_url": "URL_TO_AUDIO"
        }
        ```
    - For noise detection:
        ```json
        {
            "key": "noise",
            "audio_url": "URL_TO_AUDIO"
        }
        ```
    - For image quality checking:
        ```json
        {
            "key": "img_qual",
            "img_url": "URL_TO_IMAGE"
        }
        ```
    - For language model generation:
        ```json
        {
            "key": "llm",
            "prompt": "YOUR_PROMPT"
        }
        ```

## Project Structure

- [bogo_audio.py](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/bogo_audio.py): Contains the `BogoAudio` class for audio emotion detection.
- [bogo_img.py](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/bogo_img.py): Contains the `BogoImage` class for image emotion detection.
- [bogo_qual.py](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/bogo_qual.py): Contains the `BogoQualityChecker` class for image quality checking.
- [bogo_llm.py](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/bogo_llm.py): Contains the `BogoLlm` class for language model generation.
- [noise.py](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/noise.py): Contains the `BogoNoise` class for noise detection.
- [bogo_api.py](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/bogo_api.py): Contains the Flask API implementation.
- [training.ipynb](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/training.ipynb): Jupyter notebook for training the YOLO model.
- [config.yaml](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/config.yaml): Configuration file for training the YOLO model.

## Model Training

- The process of training the YOLO model is documented in the [training.ipynb](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/training.ipynb) notebook.
- Training parameters are configured in the [config.yaml](https://github.com/PanPeryskop/Bogomotion-2.0-API/blob/main/config.yaml) file.
- The final model was trained on a dataset of images using the YOLOv11 medium model.
- The model was trained for 300 epochs.