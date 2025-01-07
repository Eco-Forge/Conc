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

@app.post("/predict")
async def predict(
    request: Request,
    age: float = Form(...),
    cement: float = Form(...),
    water: float = Form(...),
    fa: float = Form(...),
    sp: float = Form(...),
    bfs: float = Form(...)
):
    """Process form input and return prediction"""
    try:
        # Create feature list in the correct order
        f_list = [age, cement, water, fa, sp, bfs]
        
        # Reshape and convert to DataFrame
        final_features = np.array(f_list).reshape(-1, 6)
        df = pd.DataFrame(final_features)
        
        # Make prediction
        prediction = model.predict(df)
        result = "%.2f" % round(prediction[0], 2)
        
        # Return the template with prediction
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction_text": f"The Concrete compressive strength is {result} MPa"
            }
        )
    except Exception as e:
        # Handle errors gracefully
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "prediction_text": f"Error in prediction: {str(e)}"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
