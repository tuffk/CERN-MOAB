import os
from pprint import pprint as pp
from clarifai.rest import ClarifaiApp

app = ClarifaiApp(api_key='6a5dcec33cc04be992d64746bafad339')
dirname = os.getcwd() + "/radioactive_waste"

model = app.models.get("radioactive_waste")

p1 = model.predict_by_filename(dirname+"/test3.jpg")
p2 = model.predict_by_filename(dirname+"/not1.jpg")

data = p1["outputs"][0]["data"]["concepts"]
data2 = p2["outputs"][0]["data"]["concepts"]

def detect_watse(processed_img):
    for x in processed_img:
        if x["name"] == "radioactive" and x["value"] > 0.50:
            print("radioactive waste found")
            return
    print("no waste found")

detect_watse(data)
detect_watse(data2)
