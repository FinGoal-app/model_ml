import math

# Utility function for rounding up to the nearest base
def rounded_up_to_nearest(value, base=100000):
    """
    Round up the value to the nearest specified base.
    Args:
        value (float): The value to be rounded up.
        base (int): The base to round up to.
    Returns:
        int: Rounded up value.
    """
    return math.ceil(value / base) * base

# Utility function for formatting currency
def format_currency(value):
    """
    Format a number as Indonesian Rupiah.
    Args:
        value (float): The number to format.
    Returns:
        str: Formatted string in Rupiah format.
    """
    return f'Rp{value:,.0f}'.replace(',', '.')

# Postprocessing function
def postprocess_output(predicted_normalized, scaler_y):
    """
    Postprocess the model's normalized prediction.
    Args:
        predicted_normalized (np.array): Normalized prediction from the model.
        scaler_y (MinMaxScaler): Trained scaler for target de-normalization.
    Returns:
        dict: Contains 'original', 'rounded_up', and 'formatted' predictions.
    """
    # De-normalize the prediction
    predicted_original = scaler_y.inverse_transform(predicted_normalized)[0][0]

    # Round up to the nearest 100,000
    rounded_up = rounded_up_to_nearest(predicted_original)

    # Format as currency
    formatted = format_currency(rounded_up)

    return {
        'formatted': formatted
    }