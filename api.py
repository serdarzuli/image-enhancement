
from flask import  Flask , request, jsonify

from swapper import *
from blob import *

import asyncio
import time
from datetime import datetime

app1 = Flask(__name__)


@app1.route('/api')
def four_face_swapping():

    target_pic = request.args.get("target_pic")
    main_pic = request.args.get("main_pic")
    
    response_for_url =  process(target_pic,main_pic)

    response = upload_blob(payload_url,response_for_url)
    return response

if __name__ == "__main__":
    app1.run()
    