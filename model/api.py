from fastapi import FastAPI
import dill
import  os
import pandas as pd
from pydantic import BaseModel

class Form(BaseModel):
    client_id: object
    utm_source: object
    utm_medium: object
    utm_campaign: object
    utm_adcontent: object
    device_category: object
    device_os: object
    device_brand: object
    device_model: object
    device_screen_resolution: object
    device_browser: object
    geo_country: object
    geo_city: object

class Prediction(BaseModel):
    client_id: object
    target_action: int

app = FastAPI()
# Load the model
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'target_action.pkl')
with open(file_path, 'rb') as file:
    model = dill.load(file)

@app.get('/version')
def get_version():
    return model['metadata']  # Assuming this contains version info

@app.get('/status')
def get_status():
    return 'I\'m ok'  # Escape the single quote in a string

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame([{
        'client_id' : form.client_id,
        'utm_source': form.utm_source,
        'utm_medium': form.utm_medium,
        'utm_campaign': form.utm_campaign,
        'utm_adcontent': form.utm_adcontent,
        'device_category': form.device_category,
        'device_os': form.device_os,
        'device_brand': form.device_brand,
        'device_model': form.device_model,
        'device_screen_resolution': form.device_screen_resolution,
        'device_browser': form.device_browser,
        'geo_country': form.geo_country,
        'geo_city': form.geo_city
    }])
    df = df.drop(['client_id', 'geo_city'], axis=1)
    df['device_screen_resolution'] = df['device_screen_resolution'].apply(lambda x: 0.0 if x == '(not set)' else float(x.replace('x','.' )))
    y = model['model'].predict(df)
    return{
         'client_id': form.client_id,
         'target_action' : (y[0])
    }