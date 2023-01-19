from fastapi import FastAPI, File
from starlette.responses import Response
import io
from PIL import Image
import json
from fastapi.middleware.cors import CORSMiddleware
import torch
import cv2
import numpy as np
import face_recognition
import os
from pytz import timezone
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from pydantic import BaseModel


cred = credentials.Certificate('netra-attendance-e8411b4ff5db.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

def findEncoding(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


path='known'
images=[]
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
encodeListKnown = findEncoding(images)
print("Encoding done",len(encodeListKnown))



def get_detection_from_bytes(binary_image, max_size=1024):
    input_image = np.asarray(Image.open(io.BytesIO(binary_image)))
    small_frame = cv2.resize(input_image, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(encodeListKnown, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(encodeListKnown, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = classNames[best_match_index]

        face_names.append(name)
    return face_names


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



class Data(BaseModel):
    user: str

@app.post('/database-json')
def database_json(data: Data):
    print(type(data))
    dic1=get_data_database(student_map[data.user])
    return dic1


@app.post("/image-to-json")
async def detect_return_json_result(file: bytes = File(...)):
    results = get_detection_from_bytes(file)
    print(results)
    ind_time = datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d %H:%M')
    print(ind_time)
    for person in results:
        res = db.collection("students").document(student_map[person]).update({
            'datetime': ind_time
        })
        lst=db.collection("students").document(student_map[person]).get().to_dict()['attendance']
        st=lst[0]
        ind=index_map[person]%4
        index_map[person]+=1
        new_st=st[:ind]+'1'+st[ind+1:]
        lst[0]=new_st
        print(st,new_st)
        res = db.collection("students").document(student_map[person]).update({
            'attendance': lst
        })
    return {"result": results}

