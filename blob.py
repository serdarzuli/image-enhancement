from azure.storage.blob import BlobServiceClient
import datetime
import random
import cv2
from azure.storage.blob import BlobServiceClient


payload_url = {
        "container_name" : "media",
        "connection_string" : "DefaultEndpointsProtocol=https;AccountName=storagepictorprod;AccountKey=D0QHcvlImpVm3uiCySoA+Q1EzUjJLWhgrlDmIM5yF4qVvlVnYh1c4wIj4TlCWt6IAaIE7CPUeKLq+AStBpnk3w==;EndpointSuffix=core.windows.net"
        }


    
def upload_blob(payload, file):
    container_name = payload["container_name"]
    connection_string = payload["connection_string"]
    random_str = "".join(
        random.choices(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8
        )
    )
    unix_timestamp = int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds())
    blob_name = f"{unix_timestamp}{random_str}"
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(blob_name)
    success, encoded_image = cv2.imencode(".jpeg", file)
    content_bytes = encoded_image.tobytes()
    blob_client.upload_blob(content_bytes, blob_type="BlockBlob")
    blob_properties = blob_client.get_blob_properties()
    blob_properties.content_settings.content_type = "image/jpeg"
    blob_client.set_http_headers(blob_properties.content_settings)
    blob_url = blob_client.url
    return blob_url