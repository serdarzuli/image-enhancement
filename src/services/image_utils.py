import cv2
import base64
import requests
import numpy as np
from flask import Flask
from src.services.face_restore import FaceRestorer

app = Flask(__name__)

# FaceRestorer sınıfını başlat
restorer = FaceRestorer()

def _download_image(image_url):
    """Download image and convert to numpy array."""
    image_bytes = base64.b64encode(requests.get(image_url).content)
    image_array = np.frombuffer(base64.b64decode(image_bytes), dtype=np.uint8)
    return image_array

def process_image(image_url, upscale=2, background_enhance=True, face_upsample=True):
    image_array = _download_image(image_url) 
    image = cv2.imdecode(image_array, flags=cv2.IMREAD_COLOR)

    restored_image = restorer.restore_face(
        image_path=image,
        upscale_factor=upscale,
        background_enhance=background_enhance,
        face_upsample=face_upsample
    )

    if restored_image is None:
        raise ValueError("Face restoration failed. No face detected.")

    return restored_image
