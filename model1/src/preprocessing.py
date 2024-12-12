import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load scaler objects
def load_scalers(scaler_X_path, scaler_y_path):
    import joblib
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    return scaler_X, scaler_y

# Preprocessing function
def preprocess_input(data, scaler_X):
    """
    Preprocess user input data for the financial goal model.
    Args:
        data (dict): User input data with keys: goal_amount, goal_duration, current_savings.
        scaler_X (MinMaxScaler): Trained scaler for feature normalization.
    Returns:
        np.array: Preprocessed and normalized data ready for prediction.
    """
    # Ensure all required fields are present
    required_fields = ['goal_amount', 'goal_duration', 'current_savings']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Prepare input array
    data_array = np.array([
        data['goal_amount'],
        data['goal_duration'],
        data['current_savings']
    ]).reshape(1, -1)

    # Normalize input data
    normalized_data = scaler_X.transform(data_array)
    return normalized_data