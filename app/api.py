from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="混凝土强度预测系统")

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
        print(f"接收到的数值: 龄期={age}, 水泥={cement}, 水={water}, 粉煤灰={fly_ash}, "
              f"高效减水剂={superplasticizer}, 矿渣粉={blast_furnace_slag}")
        
        # Create DataFrame with named columns in the correct order
        input_data = pd.DataFrame([[age, cement, water, fly_ash, superplasticizer, blast_furnace_slag]], 
                                columns=['age', 'cement', 'water', 'fly_ash', 'superplasticizer', 'blast_furnace_slag'])
        
        # Debug print
        print("输入数据:", input_data)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Debug print
        print(f"预测结果: {prediction[0]:.2f} MPa")
        
        result = templates.TemplateResponse("index.html", 
                                       {"request": request, 
                                        "prediction_text": f"预测的混凝土抗压强度: {prediction[0]:.2f} MPa"})
        # Debug print
        print("返回预测结果:", result.context["prediction_text"])
        return result
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        return templates.TemplateResponse("index.html", 
                                       {"request": request, 
                                        "prediction_text": f"预测过程中出现错误: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
