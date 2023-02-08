from fastapi import FastAPI, File, UploadFile
from fastapi import Request
from starlette.responses import Response
import io
from PIL import Image
from typing import List
import json
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import face_recognition
import os
from pytz import timezone
from fastapi.encoders import jsonable_encoder
from datetime import datetime
# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import firestore
from pydantic import BaseModel
import base64

# cred = credentials.Certificate('netra-attendance-e8411b4ff5db.json')
# firebase_admin.initialize_app(cred)
# db = firestore.client()

def findEncoding(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown=[]
classNames = []

def load_model():
    global encodeListKnown 
    encodeListKnown.clear()
    classNames.clear()
    print("loading model...........")
    path='known'
    images=[]
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)
    
    encodeListKnown = findEncoding(images)
    print("Encoding done",len(encodeListKnown))




def get_data_database(id):
    doc = db.collection('students').document(id)
    res = doc.get().to_dict()
    print(res)
    return res


app = FastAPI(
    title="face recognition api",
    description="face recognition + firebase connection",
    version="0.0.1",
)

origins = [
    "http://localhost",
    "http://localhost:8000",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


student_map={"sanjuth":'PK0hn9sm7iJQwYeDtGmf','saiki':'jkwlB44fgclXgTFO5dIV','shiva':'g9puYivoFWyY69vdAqJZ'}
index_map={"sanjuth":0,'saiki':0,'shiva':0}


# routes
@app.get('/notify/v1/health')
def get_health():
    return dict(msg='i am alive')


class Person(BaseModel):
    person: str

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
# ===================================================================

@app.get("/get-encodings")
async def get_encodings():
    load_model()
    return {"faces": classNames,"encodings": json.dumps(encodeListKnown,cls=NumpyEncoder)}

@app.post("/new-student")
async def new_student(file: UploadFile = File(...)):
    f=await file.read()
    input_image = Image.open(io.BytesIO(f)).convert("RGB")
    input_image.save('known/'+file.filename)
    return {
        "res":"ok"
    }

# @app.get("/get-image")
# async def get_image(name: str):
    # with open("known/"+name+".jpeg", "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read())
    # print(encoded_string.decode('utf-8'))
    # return{
    #     "image": encoded_string
    # }
    # with open("known/"+name+'.jpeg', 'rb') as open_file:
    #     byte_content = open_file.read()
    #     base64_bytes = base64.b64encode(byte_content)
    #     base64_string = base64_bytes.decode('utf-8')
    #     raw_data = {"image": base64_string}
    # #     json_data = json.dumps(raw_data, indent=2)
    #     print(json_data)
    # return Response(content=json_data, media_type="image/jpeg")
    # with open("known/"+name+)

