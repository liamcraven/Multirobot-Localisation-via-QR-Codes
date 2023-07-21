import numpy as np
import json
import os
import pickle


MULTIPLE = 32

def generate_linearization_point():
    """
    This function generates a random linearization point of 64 bit floats.
    """
    # Generate a random linearization point
    linearization_point = np.random.rand(3, 1).flatten() * MULTIPLE
    return linearization_point.tolist()

def generate_message_mean():
    """
    This function generates a random message mean of 64 bit floats.
    """
    # Generate a random message mean
    message_mean = np.random.rand(3, 1).flatten() * MULTIPLE
    return message_mean.tolist()

def generate_message_precision():
    """
    This function generates a random message precision of 64 bit floats.
    """
    # Generate a random message precision
    message_precision = np.random.rand(3, 3)
    message_precision = np.dot(message_precision, message_precision.transpose()).flatten() * MULTIPLE
    return message_precision.tolist()

def generate_example_window(linearization_points=10, factor_to_variable_messages=10, variable_to_factor_messages=10):
    """
    This function generates an example window of 64 bit floats.
    """
    # Generate a random window
    window = {}
    window["linearization_points"] = [generate_linearization_point() for _ in range(linearization_points)]
    window["factor_to_variable_messages"] = {}
    window["factor_to_variable_messages"]["message means"] = [[generate_message_mean(), generate_message_mean()] for _ in range(factor_to_variable_messages)]
    window["factor_to_variable_messages"]["message precisions"] = [[generate_message_precision(), generate_message_precision()] for _ in range(factor_to_variable_messages)]
    window["variable_to_factor_messages"] = {}
    window["variable_to_factor_messages"]["message means"] = [[generate_message_mean(), generate_message_mean()] for _ in range(variable_to_factor_messages)]
    window["variable_to_factor_messages"]["message precisions"] = [[generate_message_precision(), generate_message_precision()] for _ in range(variable_to_factor_messages)]
    
    return window

def main():
    WINDOW_SIZE = 15
    #Remove old file
    if os.path.exists(f"example_window_{WINDOW_SIZE}.json"):
        os.remove(f"example_window_{WINDOW_SIZE}.json")
    # Generate a random window
    window = generate_example_window(WINDOW_SIZE, WINDOW_SIZE, WINDOW_SIZE)
    # Save the window
    json_str = json.dumps(window)
    with open(f"example_window_{WINDOW_SIZE}.json", "w") as f:
        f.write(json_str)


if __name__ == "__main__":
    main()