import sys
import os
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

CODEFORMER_PATH = os.path.abspath("./CodeFormer/CodeFormer")
if CODEFORMER_PATH not in sys.path:
    sys.path.append(CODEFORMER_PATH)

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY

class FaceRestorer:
    PRETRAIN_MODEL_URLS = {
        'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
        'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
    }
    

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.upsampler = self.initialize_realesrgan()
        self.codeformer_net = self.load_codeformer_model()
        self.face_helper = FaceRestoreHelper(
            upscale=2, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50", save_ext="png", use_parse=True
        )
    

    def check_and_download_weights(self):
        """Dwonaload missing files"""
        weights_dir = "CodeFormer/CodeFormer/weights"
        for model_name, url in self.PRETRAIN_MODEL_URLS.items():
            model_path = os.path.join(weights_dir, model_name)
            if not os.path.exists(model_path):
                load_file_from_url(url, model_dir=model_path, progress=True)


    def initialiaze_realesrgan():
        half = torch.cuda.is_available()
        model = RRDBNet(num_in_ch=3,num_out_ch=3,num_feat=64,num_block=23,num_grow_ch=32,scale=2)
        return RealESRGANer(
            scale=2,
            model_path="CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
            model=model,
            tile=400,
            tile_pad=40,
            pre_pad=0,
            half=half,
        )
    

    def load_codeformer_model(self):
        """CodeFormer modelini yükler."""
        model_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
        model = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512, codebook_size=1024, n_head=8, n_layer=9, connect_list=["32", "64", "128", "256"])
        model.load_state_dict(torch.load(model_path, map_location=self.device)["params_ema"])
        model.to(self.device).eval()
        return model


    def restore_face(self, image_path, upscale_factor=2, background_enhance=True, face_upsample=True):
        """face restore and made upscale"""
        image = cv2.imread(image_path)
        self.face_helper.read_image(image)
        
        num_faces = self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        if num_faces == 0:
            print("No face detected in image.")
            return None
        
        self.face_helper.align_warp_face()
        restored_faces = []

        for cropped_face in self.face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.codeformer_net(cropped_face_t, w=0.7, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                restored_faces.append(restored_face.astype("uint8"))
        
        self.face_helper.add_restored_face(restored_faces[0])
        if background_enhance:
            bg_img = self.upsampler.enhance(image, outscale=upscale_factor)[0]
        else:
            bg_img = None
        
        restored_image = self.face_helper.paste_faces_to_input_image(
            upsample_img=bg_img, draw_box=False, face_upsampler=self.upsampler if face_upsample else None
        )

        return restored_image
   