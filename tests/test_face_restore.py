import pytest
import torch
import numpy as np
import cv2
from src.restoration import check_ckpts, set_realesrgan, face_restoration

@pytest.mark.skip(reason="Test the model files")
def test_check_ckpts():
    try:
        check_ckpts()
    except Exception as e:
        pytest.fail(f"Model dosyaları eksik veya indirme başarısız: {e}")

def test_face_restoration(mocker):
    img = np.zeros((150, 150, 3), dtype=np.uint8)

    # start the model
    upsampler = set_realesrgan()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class MockModel:
        def __call__(self, x, w, adain):
            return x 

    codeformer_net = MockModel()

    mocker.patch("src.restoration.FaceRestoreHelper.get_face_landmarks_5", return_value=1)
    mocker.patch("src.restoration.FaceRestoreHelper.align_warp_face", return_value=None)

    result_img = face_restoration(
        img, background_enhance=False, face_upsample=False,
        upscale=1, codeformer_fidelity=0.5,
        upsampler=upsampler, codeformer_net=codeformer_net, device=device
    )

    assert isinstance(result_img, np.ndarray)
    assert result_img.shape == (150, 150, 3)
