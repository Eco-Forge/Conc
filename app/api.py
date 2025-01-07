from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="Concrete Strength Predictor")

# Mount templates directory
templates = Jinja2Templates(directory="app/templates")

# Load the model
model = joblib.load("XGBoost_Regressor_model.pkl")

@app.get("/")
async def home(request: Request):
    """Render the home page with the prediction form"""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction_text": ""}
    )

# ... existing code ...

@app.post("/predict")
async def predict(request: Request, 
                 age: float = Form(...),
                 cement: float = Form(...),
                 water: float = Form(...),
                 fly_ash: float = Form(...),
                 superplasticizer: float = Form(...),
                 blast_furnace_slag: float = Form(...)):
    try:
        # Debug print
        print(f"Received values: age={age}, cement={cement}, water={water}, fly_ash={fly_ash}, "
              f"superplasticizer={superplasticizer}, blast_furnace_slag={blast_furnace_slag}")
        
        # Create DataFrame with named columns in the correct order
        input_data = pd.DataFrame([[age, cement, water, fly_ash, superplasticizer, blast_furnace_slag]], 
                                columns=['age', 'cement', 'water', 'fly_ash', 'superplasticizer', 'blast_furnace_slag'])
        
        # Debug print
        print("Input data:", input_data)
        
        # Make prediction
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
