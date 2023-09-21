"""
This project is developed by Haofan Wang to support face swap in single frame. Multi-frame will be supported soon!

It is highly built on the top of insightface, sd-webui-roop and CodeFormer.
"""
import cv2
import argparse
import insightface
import requests
import requests
import insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import base64
from flask import  Flask , request, jsonify
import api
from api import *
import numpy as np
from numpy import asarray
from typing import List, Union, Dict, Set, Tuple
from restoration import *
from blob import *
app1 = Flask(__name__)

assert float(".".join(insightface.__version__.split(".")[:2])) >= float("0.7")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))
# swap part
swapper = insightface.model_zoo.get_model(
    "inswapper_128.onnx", download=True, download_zip=True
)

    
def _save_picture(picture):
    save_picture = base64.b64encode(requests.get(picture).content)
    im_bytes = base64.b64decode(save_picture)
    im_arr = np.frombuffer(
        im_bytes, dtype=np.uint8
    ) 
    return im_arr

def process(target_pic,main_pic):
    im_arr = _save_picture(target_pic)  # input target_picture base64
     # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    # Detected faces
    faces = app.get(img)
    fig, axs = plt.subplots(1, 4, figsize=(12, 6))
    for i, face in enumerate(faces):
        bbox = face["bbox"]
        bbox = [int(b) for b in bbox]
        axs[i].imshow(img[bbox[1] : bbox[3], bbox[0] : bbox[2], ::-1])
        axs[i].axis("off")
    # Sorts left to right
    faces = sorted(faces, key=lambda x: x.bbox[0])
    res = img.copy()
    assert len(faces) == 4  # Confirm 4 faces found
    source_face = faces[2]
    bbox = source_face["bbox"]
    bbox = [int(b) for b in bbox]
    plt.imshow(img[bbox[1] : bbox[3], bbox[0] : bbox[2], ::-1])
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)
    res = []
    for face in faces:
        _img, _ = swapper.get(img, face, source_face, paste_back=False)
        res.append(_img)
    res = np.concatenate(res, axis=1)
    # 1. Detect and Save my Face
    im_arr = _save_picture(main_pic)  # input main picture base64
    main_picture = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    main_picture_faces = app.get(main_picture)
    assert len(main_picture_faces) == 1
    main_picture_face = main_picture_faces[0]
    bbox = main_picture_face["bbox"]
    bbox = [int(b) for b in bbox]
    plt.imshow(main_picture[bbox[1] : bbox[3], bbox[0] : bbox[2], ::-1])
    # 2. Detect Friend's faces
    faces = app.get(img)  # First image
    # 3. Swap my face for theirs on the image
    res = img.copy()
    for face in faces:
        res = swapper.get(res, face, main_picture_face, paste_back=True)
    
    args = parse_args()

    check_ckpts()
    
    # https://huggingface.co/spaces/sczhou/CodeFormer
    upsampler = set_realesrgan()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                     codebook_size=1024,
                                                     n_head=8,
                                                     n_layers=9,
                                                     connect_list=["32", "64", "128", "256"],
                                                    ).to(device)
    ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
    checkpoint = torch.load(ckpt_path)["params_ema"]
    codeformer_net.load_state_dict(checkpoint)
    codeformer_net.eval()
    result_image = face_restoration(res, 
                                    args.background_enhance, 
                                    args.face_upsample, 
                                    args.upscale, 
                                    args.codeformer_fidelity,
                                    upsampler,
                                    codeformer_net,
                                    device)
    result_image = asarray(result_image)

    print("Result saved successfully")
    return result_image






def parse_args():
    parser = argparse.ArgumentParser(description="Face swap.")
    # parser.add_argument("--source_img", type=str, required=True, help="The path of source image, it can be multiple images, dir;dir2;dir3.")
    # parser.add_argument("--target_img", type=str, required=True, help="The path of target image.")
    parser.add_argument("--output_img", type=str, required=False, default="result1.jpg", help="The path and filename of output image.")
    # parser.add_argument("--source_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to use (left to right) in the source image, starting at 0 (-1 uses all faces in the source image")
    # parser.add_argument("--target_indexes", type=str, required=False, default="-1", help="Comma separated list of the face indexes to swap (left to right) in the target image, starting at 0 (-1 swaps all faces in the target image")
    parser.add_argument("--face_restore", action="store_true", help="The flag for face restoration.")
    parser.add_argument("--background_enhance", action="store_true", help="The flag for background enhancement.")
    parser.add_argument("--face_upsample", action="store_true", help="The flag for face upsample.")
    parser.add_argument("--upscale", type=int, default=1, help="The upscale value, up to 4.")
    parser.add_argument("--codeformer_fidelity", type=float, default=0.5, help="The codeformer fidelity.")
    args = parser.parse_args()
    return args

    



