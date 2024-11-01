from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load the model
model = joblib.load('server/flight_delay_model.pkl')
        
# Load the airport data
airports = pd.read_csv('data/airports.csv')

class PredictionRequest(BaseModel):
    day_of_week: int
    airport_id: int

@app.get("/predict")
# 1=Monday, 13930=ORD
# 1=Monday, 12892=LAS
def predict(day_of_week: int, airport_id: int):
    try:
        # Prepare the input data for the model
        input_data = [[day_of_week, airport_id]]

        # Make the prediction
        print('Confidence: ', model.predict_proba(input_data))
        print('Prediction: ', model.predict(input_data))
        
        confidence = model.predict_proba(input_data).max()
        prediction = int(model.predict(input_data)[0])

        return {
            "delay_chance": prediction,
            # round to whole number
            "confidence": round(confidence * 100)            
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/airports")
def get_airports():
    try:
        sorted_airports = airports.sort_values(by='AirportName')
        return sorted_airports.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)