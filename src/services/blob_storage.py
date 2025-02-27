import datetime
import random
import cv2
from azure.storage.blob import BlobServiceClient
from config import CONTAINER_NAME, CONNECTING_STRING

# 🔹 Constants
RANDOM_STRING_LENGTH = 8  

def _generate_unique_filename():
    """Random name file."""
    random_str = "".join(
        random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=RANDOM_STRING_LENGTH)
    )
    unix_timestamp = int(datetime.datetime.utcnow().timestamp())  
    return f"{unix_timestamp}_{random_str}.jpeg"

def _convert_image_to_bytes(image):
    success, encoded_image = cv2.imencode(".jpeg", image)
    if not success:
        raise ValueError("Failed to encode the image.")
    return encoded_image.tobytes()

def upload_blob(image):
    try:
        blob_name = _generate_unique_filename()
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTING_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        blob_client = container_client.get_blob_client(blob_name)

        content_bytes = _convert_image_to_bytes(image)
        blob_client.upload_blob(content_bytes, blob_type="BlockBlob")

        blob_client.set_http_headers({"content-type": "image/jpeg"})

        return blob_client.url  
    except Exception as e:
        print(f"Error: Failed to upload blob! Details: {e}")
        return None
