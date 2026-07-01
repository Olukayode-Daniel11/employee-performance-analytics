
import pandas as pd

from app.utils import load_object

# Load Artifacts

# Load Artifacts

model = load_object("app/model.pkl")
preprocessor = load_object("app/preprocessor.pkl")



def predict_performance(input_data: dict):
    """
    Predict employee performance.
    """

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Preprocess input
    processed_data = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(processed_data)

    return int(prediction[0])

