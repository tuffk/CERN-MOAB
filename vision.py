# Pip install the client:
# pip install clarifai
import os
from pprint import pprint as pp
from clarifai.rest import ClarifaiApp

# Create your API key in your account's Application details page:
# https://clarifai.com/apps

app = ClarifaiApp(api_key='6a5dcec33cc04be992d64746bafad339')
# You can also create an environment variable called `CLARIFAI_API_KEY`
# and set its value to your API key.
# In this case, the construction of the object requires no `api_key` argument.

pwd = os.path.dirname(os.path.realpath(__file__))
model = app.public_models.general_model
response = model.predict_by_filename(pwd+"radioactive_waste/table.jpg", max_concepts=20)
data = response['outputs'][0]['data']['concepts']
for x in data:
    print(f"{x['name']}: {x['value']:10.3}")
print("")
