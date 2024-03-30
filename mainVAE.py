import os
import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from preprocessing import preprocess_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from tabulate import tabulate
from VariationalBGMM import VariationalBGMM  # Import your custom VGMM class
from VariationalAutoencoder import VariationalAutoencoder  # Import your VAE class

# Silence the joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Set this to the number of cores you want to use

# Suppress the sklearn warning
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to calculate purity
def purity_score(y_true, y_pred):
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    contingency_matrix = np.zeros((2, 2))
    for i in range(len(y_true)):
        contingency_matrix[y_true[i], y_pred[i]] += 1
    purity = np.sum(np.max(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    return purity

# Function to calculate Rand Index
def rand_index(y_true, y_pred):
    a = b = c = d = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j] and y_pred[i] == y_pred[j]:
                a += 1
            elif y_true[i] != y_true[j] and y_pred[i] != y_pred[j]:
                b += 1
            elif y_true[i] == y_true[j] and y_pred[i] != y_pred[j]:
                c += 1
            elif y_true[i] != y_true[j] and y_pred[i] == y_pred[j]:
                d += 1
    rand_index = (a + b) / (a + b + c + d)
    return rand_index

# Local directory for CSV files
csv_directory = os.path.join(os.getcwd(), "file_csv")

# Load and preprocess the data
for file_name in os.listdir(csv_directory):
    if file_name.endswith(".csv"):
        file_path = os.path.join(csv_directory, file_name)
        # Preprocessing
        data = pd.read_csv(file_path)
        X, y = preprocess_data(data)  # Assuming preprocess_data function is defined
        
        # VAE for dimensionality reduction
        vae = VariationalAutoencoder(input_dim=X.shape[1], latent_dim=2)  # Assuming VAE is correctly implemented
        X_reduced = vae.fit_transform(X)
        
        # Scaling the reduced data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_reduced)
        
        # Clustering with KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans_clusters = kmeans.fit_predict(X_scaled)
        kmeans_purity = purity_score(y, kmeans_clusters)
        kmeans_rand_index = rand_index(y, kmeans_clusters)
        
        # Clustering with Gaussian Mixture Model
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm_clusters = gmm.fit_predict(X_scaled)
        gmm_purity = purity_score(y, gmm_clusters)
        gmm_rand_index = rand_index(y, gmm_clusters)
        
        # Clustering with Variational Gaussian Mixture Model (VBGMM)
        vbgmm = VariationalBGMM(n_components=2, random_state=42)  # Assuming your VGMM class is correctly implemented
        vbgmm_clusters = vbgmm.fit_predict(X_scaled)
        vbgmm_purity = purity_score(y, vbgmm_clusters)
        vbgmm_rand_index = rand_index(y, vbgmm_clusters)
        
        # Output results
        print(f"Results for file: {file_name}")
        headers = ["Model", "Purity", "Rand Index"]
        table_data = [
            ["KMeans", kmeans_purity, kmeans_rand_index],
            ["Gaussian Mixture Model", gmm_purity, gmm_rand_index],
            ["Variational Gaussian Mixture Model", vbgmm_purity, vbgmm_rand_index]
        ]
        print(tabulate(table_data, headers=headers))
