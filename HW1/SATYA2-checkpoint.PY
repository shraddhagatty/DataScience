import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
 #Step 1: Read the data
data_path = "/Users/ravishanker/Documents/GITHUB/DataScience"  # Update this with your actual data path
data = pd.read_csv(os.path.join(data_path, "test_sample.csv"))
 
# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop("Y", axis=1))
 
# Step 1: Apply PCA and Find n_PCA
n_PCA = None
explained_variance_ratio_threshold = 0.9  # Adjust this threshold as needed
 
pca = PCA()
pca_features = pca.fit_transform(scaled_data)
 
for n_components in range(1, len(pca.explained_variance_ratio_)):
    explained_variance_ratio = np.sum(pca.explained_variance_ratio_[:n_components])
    if explained_variance_ratio >= explained_variance_ratio_threshold:
        n_PCA = n_components
        break
 
# Step 2: Fit Linear Regression Model
selected_features = pca_features[:, :n_PCA]
model = LinearRegression()
model.fit(selected_features, data["Y"])
 
# Step 3: Calculate Model Dimensionality Reduction
original_feature_count = len(data.columns) - 1  # Subtract target variable "Y"
model_dimensionality_reduction = original_feature_count - n_PCA
 
# Step 4: Model Evaluation
r_squared = model.score(selected_features, data["Y"])
 
# Print results
print(f"Model dimensionality reduction: {model_dimensionality_reduction}")
print(f"Determination coefficient (R-squared): {r_squared}")
 