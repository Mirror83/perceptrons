import pandas as pd
import numpy as np

from perceptron import Perceptron

"""
The array below represents all configurations that a 2x2 image in which the 
squares can be either "dark" or "light" can be in.

The first value in each row represents the unit input for the bias
The values from [1:] in each row represent the intensity of each of the four squares
in the 2x2 image in the following order:
top-right, top-left, bottom-right, bottom-left.

1 represents "dark", and 0 represents "light".
"""
inputs = np.array([
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 0],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
], dtype=np.bool)

targets = np.zeros(inputs.shape[0], dtype=np.bool)

# Here, we populate the targets based on the following rule:
# if three or more of the squares are dark, then the whole image
# is dark, otherwise it is light
for i, input_vector in enumerate(inputs):
    targets[i] = 1 if sum(input_vector[1:]) >= 3 else 0

df = pd.DataFrame(
    inputs,
    columns=["Threshold", "Top-left", "Top-right",
             "Bottom-left", "Bottom-right"],
    dtype=np.bool)
df["results"] = targets
print(df)

model = Perceptron(inputs, targets).train()
predictions = model.predict()

print(f"Correct prediction: {np.all(predictions == targets)}")
