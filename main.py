# 1. Library imports
import uvicorn
from fastapi import FastAPI, Request
import joblib
import pandas as pd
import shap
import json
pd.options.mode.chained_assignment = None  # default='warn'

# 2. Create the app object
app = FastAPI()

# Chargement du modèle
model = joblib.load('model.pkl')
data = joblib.load('sample_test_set.pickle')

@app.post('/predict/')
async def predict(request: Request):
    json_ = request.json()
    query = pd.DataFrame(json_)
    prediction = model.predict(query.values)
    return int(prediction)

@app.get('/shap_values/{client_id}')
async def shap_values(client_id : int):
    data_unique = data[data.index == client_id]
    df_preprocess = model.named_steps['preprocessor'].transform(data_unique)
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer.shap_values(df_preprocess)
    shap_client = json.dumps(shap_values.tolist())
    return shap_client

# 5. Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='35.180.29.152', port=8000, reload=True)