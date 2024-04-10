import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Replace with your actual data path
data_path = "/Users/ravishanker/Documents/GITHUB/DataScience"
data = pd.read_csv(os.path.join(data_path, "test_sample.csv"))

# Read data
data = pd.read_csv(os.path.join(data_path, "test_sample.csv"))

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop("Y", axis=1))

# Step 1: Fit Linear Regression Models and Find n_orig
n_orig = None
for j in range(2, 492):
    cols = [f"X{i}" for i in range(1, j+1)]
    model = LinearRegression()
    model.fit(data[cols], data["Y"])
    r_squared = model.score(data[cols], data["Y"])
    if r_squared > 0.9:
        n_orig = j
        break
print(f"nii {n_orig}")
# Step 2: Apply PCA and Find n_PCA
pca = PCA()
pca_features = pca.fit_transform(scaled_data)

# Compute relative importance measures
pca_importance = np.abs(pca.components_)

# Compute the cumulative sum of relative importance measures
pca_importance_sum = np.sum(pca_importance, axis=1)

# Order the indices based on relative importance
ordered_indices = np.argsort(pca_importance_sum)[::-1]

# Reorder the PCA features based on importance
ordered_features = pca_features[:, ordered_indices]

# Iterate through the cumulative sum and find the index where it first exceeds or equals 0.9
n_PCA = np.where(np.cumsum(pca_importance_sum[ordered_indices]) >= 0.9)[0][0] + 1

print("Smallest number of PCA factors with R-squared >= 0.9 (reordered by importance):", n_PCA)