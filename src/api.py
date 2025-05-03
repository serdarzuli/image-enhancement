from flask import Flask, request
from src.services.blob_storage import upload_blob
from src.services.image_utils import process_image

app = Flask(__name__)

@app.route('/api')
def image_enhancement():
    picture = request.args.get("target_pic")
    if not picture or not picture.startswith(('http://', 'https://')):
        return {"error": "Invalid or missing image URL"}, 400

    try:
        quality_up = process_image(picture)
        response = upload_blob(quality_up)
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}, 500

if __name__ == "__main__":
    app.run()
    