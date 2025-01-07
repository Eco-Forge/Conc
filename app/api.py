from fastapi import FastAPI, Request, Form, Cookie
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="混凝土强度预测系统")

# Mount templates directory
templates = Jinja2Templates(directory="app/templates")

# Load the model
model = joblib.load("XGBoost_Regressor_model.pkl")

# Add translations
TRANSLATIONS = {
    'en': {
        'title': 'Concrete Compressive Strength Predictor',
        'navbar_title': 'Concrete Strength Predictor',
        'form_title': 'Predict Concrete Strength',
        'age': 'Age (in days)',
        'cement': 'Cement (in kg)',
        'water': 'Water (in kg)',
        'fly_ash': 'Fly Ash (in kg)',
        'superplasticizer': 'Superplasticizer (in kg)',
        'blast_furnace_slag': 'Blast Furnace Slag (in kg)',
        'predict_button': 'Predict Strength',
        'prediction_prefix': 'Predicted Concrete Strength: ',
        'enter_age': 'Enter concrete age',
        'enter_cement': 'Enter cement quantity',
        'enter_water': 'Enter water quantity',
        'enter_fly_ash': 'Enter fly ash quantity',
        'enter_superplasticizer': 'Enter superplasticizer quantity',
        'enter_blast_furnace_slag': 'Enter blast furnace slag quantity'
    },
    'zh': {
        'title': '混凝土抗压强度预测器',
        'navbar_title': '混凝土强度预测器',
        'form_title': '预测混凝土强度',
        'age': '龄期（天）',
        'cement': '水泥（千克）',
        'water': '水（千克）',
        'fly_ash': '粉煤灰（千克）',
        'superplasticizer': '高效减水剂（千克）',
        'blast_furnace_slag': '矿渣（千克）',
        'predict_button': '强度预测',
        'prediction_prefix': '混凝土抗压强度为：',
        'enter_age': '请输入混凝土龄期',
        'enter_cement': '请输入水泥用量',
        'enter_water': '请输入水用量',
        'enter_fly_ash': '请输入粉煤灰用量',
        'enter_superplasticizer': '请输入减水剂用量',
        'enter_blast_furnace_slag': '请输入矿渣用量'
    }
}

@app.get("/")
async def home(request: Request, lang: Optional[str] = Cookie(default="en")):
    """Render the home page with the prediction form"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request, 
            "prediction_text": "",
            "translations": TRANSLATIONS[lang],
            "current_lang": lang
        }
    )

@app.get("/switch-language/{lang}")
async def switch_language(lang: str):
    response = RedirectResponse(url="/")
    response.set_cookie(key="lang", value=lang)
    return response

@app.post("/predict")
async def predict(
    request: Request,
    lang: Optional[str] = Cookie(default="en"),
    age: float = Form(...),
    cement: float = Form(...),
    water: float = Form(...),
    fly_ash: float = Form(...),
    superplasticizer: float = Form(...),
    blast_furnace_slag: float = Form(...)
):
    try:
        # Debug print
        print(f"Received values: age={age}, cement={cement}, water={water}, fly_ash={fly_ash}, "
              f"superplasticizer={superplasticizer}, blast_furnace_slag={blast_furnace_slag}")
        
        # Create DataFrame with named columns in the correct order
        input_data = pd.DataFrame([[age, cement, water, fly_ash, superplasticizer, blast_furnace_slag]], 
                                columns=['age', 'cement', 'water', 'fly_ash', 'superplasticizer', 'blast_furnace_slag'])
        
        # Debug print
        print("Input data:", input_data)
        
        prediction = model.predict(input_data)
        
        # Debug print
        print(f"Prediction result: {prediction[0]:.2f} MPa")
        
        result = templates.TemplateResponse("index.html", 
                                       {"request": request, 
                                        "prediction_text": f"Predicted Concrete Strength: {prediction[0]:.2f} MPa"})
        # Debug print
        print("Returning response with prediction_text:", result.context["prediction_text"])
        return result
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return templates.TemplateResponse("index.html", 
                                       {"request": request, 
                                        "prediction_text": f"Error in prediction: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
