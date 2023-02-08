import numpy as np
import requests
import json

a=np.array([1.200,2.300,3.400 ])
b=np.array([4.200,5.300,6.400 ])
face_encodings=[]
face_encodings.append(a)
face_encodings.append(b)

print(li)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

payload={"data" : face_encodings}
headers = {}
url ="http://localhost:8000/recognize-faces"
response = requests.request("POST", url, headers=headers, data=json.dumps(payload,cls=NumpyEncoder))
print(json.loads(response.text)['result'])