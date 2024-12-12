import os
from tensorflow.keras.models import load_model # type: ignore
from preprocessing import preprocess_input, load_scalers
from postprocessing import postprocess_output

# Load model and scalers
MODEL_PATH = os.path.join('..', 'model', 'financial_goal_model.h5')
SCALER_X_PATH = os.path.join('..', 'model', 'scaler_X.pkl')
SCALER_Y_PATH = os.path.join('..', 'model', 'scaler_y.pkl')

model = load_model(MODEL_PATH)
scaler_X, scaler_y = load_scalers(SCALER_X_PATH, SCALER_Y_PATH)

# Pipeline function
def financial_goal_pipeline(user_data):
    # Preprocessing
    normalized_input = preprocess_input(user_data, scaler_X)

    # Modeling (prediction)
    predicted_normalized = model.predict(normalized_input)

    # Postprocessing
    result = postprocess_output(predicted_normalized, scaler_y)

    return result