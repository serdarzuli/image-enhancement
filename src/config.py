import os
from dotenv import load_dotenv

#.env load file
load_dotenv()

CONTAINER_NAME="media"
CONNECTING_STRING= os.getenv("CONNECTING_STRING")
