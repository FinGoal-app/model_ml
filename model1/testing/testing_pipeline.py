import sys
import os

# Tambahkan folder src ke Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from pipeline import financial_goal_pipeline
from preprocessing import preprocess_input, load_scalers
from postprocessing import postprocess_output

# Example usage for testing the pipeline
if __name__ == "__main__":
    user_data = {
        'goal_amount': 14000000,
        'goal_duration': 6,
        'current_savings': 2000000
    }

    try:
        # Call the pipeline with the test data
        result = financial_goal_pipeline(user_data)
        print("Test Result:")
        print(result)
    except Exception as e:
        print("An error occurred during testing:")
        print(str(e))