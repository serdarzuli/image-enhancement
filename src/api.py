from flask import  Flask, request
from src.services.blob_storage import upload_blob
from src.services.image_utils import  process_image

app = Flask(__name__)

@app.route('/api')
def four_face_swapping():

    picture = request.args.get("target_pic")
    quality_up = process_image(picture)
    response = upload_blob(quality_up)
    return response

if __name__ == "__main__":
    app.run()
    