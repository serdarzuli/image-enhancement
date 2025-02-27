import pytest
import numpy as np
import requests
import base64
import cv2
from unittest.mock import patch
from src.services import process, _save_picture

@pytest.fixture
def fake_image_url():
    return "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/800px-PNG_transparency_demonstration_1.png"

@patch("src.services.requests.get")
def test_save_picture(mock_get, fake_image_url):
    """Resmin base64 olarak kaydedilip işlenmesini test et"""
    mock_get.return_value.content = base64.b64decode(
        base64.b64encode(b"fake_image_data")
    )

    result = _save_picture(fake_image_url)

    assert isinstance(result, np.ndarray)

@patch("src.services.face_restoration")
def test_process(mock_face_restoration, fake_image_url):
    mock_face_restoration.return_value = np.zeros((150, 150, 3), dtype=np.uint8)

    result_img = process(fake_image_url)

    assert isinstance(result_img, np.ndarray)
    assert result_img.shape == (150, 150, 3)
