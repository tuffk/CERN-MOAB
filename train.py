import os
from pprint import pprint as pp
from clarifai.rest import ClarifaiApp

app = ClarifaiApp(api_key='6a5dcec33cc04be992d64746bafad339')
"""
# import a few labelled images
app.inputs.create_image_from_url(url="https://samples.clarifai.com/dog1.jpeg", concepts=["cute dog"], not_concepts=["cute cat"])
app.inputs.create_image_from_url(url="https://samples.clarifai.com/dog2.jpeg", concepts=["cute dog"], not_concepts=["cute cat"])

app.inputs.create_image_from_url(url="https://samples.clarifai.com/cat1.jpeg", concepts=["cute cat"], not_concepts=["cute dog"])
app.inputs.create_image_from_url(url="https://samples.clarifai.com/cat2.jpeg", concepts=["cute cat"], not_concepts=["cute dog"])

model = app.models.create(model_id="pets", concepts=["cute cat", "cute dog"])

model = model.train()

# predict with samples
print (model.predict_by_url(url="https://samples.clarifai.com/dog3.jpeg"))
print (model.predict_by_url(url="https://samples.clarifai.com/cat3.jpeg"))
"""

dirname = os.getcwd() + "/radioactive_waste"
all_files = os.listdir(dirname)
yes_files = [x for x in all_files if "yes" in x]

training_imgs = []
for x in yes_files:
    app.inputs.create_image_from_filename(dirname+f"/{x}", concepts=["radioactive"], not_concepts=["other"])

# model = app.models.create(model_id="radioactive_waste", concepts=["radioactive"])
model = app.models.get("radioactive_waste")
model = model.train()


p1 = model.predict_by_filename(dirname+"/yes1.jpg")
p2 = model.predict_by_filename(dirname+"/not1.jpg")


print("YES")
pp(p1)
print("-----------------------------")
pp(p2)
