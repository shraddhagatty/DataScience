
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import os

# Step 1: Read the data
data_path = "/Users/ravishanker/Documents/GITHUB/DataScience"  # Update this with your actual data path
data = pd.read_csv(os.path.join(data_path, "test_sample.csv"))

# Step 2: Find n_orig
n_orig = None
for j in range(2, 492):
    predictors = data.iloc[:, 1:j]  # Use predictors X1 to Xj
    target = data['Y']
    model = LinearRegression().fit(predictors, target)
    r_squared = model.score(predictors, target)
    if r_squared > 0.9:
        n_orig = j
        break