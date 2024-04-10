
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

# Step 3: Apply PCA and find n_PCA
pca = PCA()
pca.fit(data.iloc[:, 1:491])  # Exclude Y column
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
n_PCA = np.argmax(cumulative_variance_ratio > 0.9) + 1

# Step 4: Calculate model dimensionality reduction
model_dimensionality_reduction = n_orig - n_PCA

# Output the results
print("Model Dimensionality Reduction:", round(model_dimensionality_reduction, 5))
print("Determination Coefficient (R-squared) for n_PCA model:", round(cumulative_variance_ratio[n_PCA - 1], 5))
