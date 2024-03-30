import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from preprocessing import preprocess_data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import f1_score
import tensorflow as tf  # Assuming you're using TensorFlow
from encoder import train_autoencoder
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

# Directory path for CSV files
dossier_csv = "D:\\recherche\\olfaBziouch\\files_csv\\file csv\\"

# Lists to store F-score, ARI, and purity for each method
f_score_list_kmeans = []
ari_list_kmeans = []
purity_list_kmeans = []

f_score_list_gmm = []
ari_list_gmm = []
purity_list_gmm = []

f_score_list_ae = []  # Modified for AE
ari_list_ae = []  # Modified for AE
purity_list_ae = []  # Modified for AE

# Loop through CSV files in the folder
for fichier in os.listdir(dossier_csv):
    if fichier.endswith(".csv"):
        chemin_fichier = os.path.join(dossier_csv, fichier)
        print(chemin_fichier)

        # Create a folder for graphs
        dossier_graphes_fichier: str = os.path.join("D:\\recherche\\olfaBziouch\\codeArticleFinal\\AE", f"{fichier.split('.')[0]}_graphes")
        if not os.path.exists(dossier_graphes_fichier):
            os.makedirs(dossier_graphes_fichier)

        # Lists to store F-score, ARI, and purity for each iteration
        f_score_list_kmeans_iter = []
        ari_list_kmeans_iter = []
        purity_list_kmeans_iter = []

        f_score_list_gmm_iter = []
        ari_list_gmm_iter = []
        purity_list_gmm_iter = []

        f_score_list_ae_iter = []  # Modified for AE
        ari_list_ae_iter = []  # Modified for AE
        purity_list_ae_iter = []  # Modified for AE
        time_list_iter = []

        X_train, X_test, y_train, y_test, label = preprocess_data(chemin_fichier, 0.2)

        iterations = list(range(150))

        for i in iterations:
            start_time = time.time()
            model = train_autoencoder(X_train, X_test, fichier, dossier_graphes_fichier, i)  # Assuming train_autoencoder returns the model
            print("Autoencoder training finished")

            # Obtain lower-dimensional representations from the encoder part of the autoencoder
            encoder = tf.keras.Model(inputs=model.input, outputs=model.layers[0].output)
            q = encoder.predict(X_test)
            print("Encoder prediction finished")

            q = q.reshape(q.shape[0], 3)
            scaler = StandardScaler()
            q_normalized = scaler.fit_transform(q)

            num_clusters = 2

            end_time = time.time()  # Record the end time
            iteration_time = end_time - start_time  # Calculate the iteration time
            time_list_iter.append(iteration_time)
            y_test_1d = y_test.values.flatten()

            kmeans = KMeans(n_clusters=num_clusters, random_state=1)
            kmeans.fit(q_normalized)
            cluster_assignments_kmeans = kmeans.predict(q_normalized)

            gmm = GaussianMixture(n_components=num_clusters, random_state=1)
            gmm.fit(q_normalized)
            cluster_assignments_gmm = gmm.predict(q_normalized)

            # Calculate and append F-score, ARI, and purity for each method
            purity_kmeans = purity_score(y_test, cluster_assignments_kmeans)
            ari_kmeans = adjusted_rand_score(y_test, cluster_assignments_kmeans)
            f_score_kmeans = f1_score(y_test, cluster_assignments_kmeans, average='weighted')

            purity_gmm = purity_score(y_test, cluster_assignments_gmm)
            ari_gmm = adjusted_rand_score(y_test, cluster_assignments_gmm)
            f_score_gmm = f1_score(y_test, cluster_assignments_gmm, average='weighted')

            # Modified for AE
            purity_ae = purity_score(y_test, cluster_assignments_kmeans)
            ari_ae = adjusted_rand_score(y_test, cluster_assignments_kmeans)
            f_score_ae = f1_score(y_test, cluster_assignments_kmeans, average='weighted')

            if f_score_kmeans < 0.5:
                f_score_kmeans = f1_score(y_test.replace({0: 1, 1: 0}), cluster_assignments_kmeans, average='weighted')
            if f_score_gmm < 0.5:
                f_score_gmm = f1_score(y_test.replace({0: 1, 1: 0}), cluster_assignments_gmm, average='weighted')
            if f_score_ae < 0.5:  # Modified for AE
                f_score_ae = f1_score(y_test.replace({0: 1, 1: 0}), cluster_assignments_kmeans, average='weighted')
            if ari_kmeans < 0.5:
                ari_kmeans = adjusted_rand_score(y_test.replace({0: 1, 1: 0}), cluster_assignments_kmeans)
            if ari_gmm < 0.5:
                ari_gmm = adjusted_rand_score(y_test.replace({0: 1, 1: 0}), cluster_assignments_gmm)
            if ari_ae < 0.5:  # Modified for AE
                ari_ae = adjusted_rand_score(y_test.replace({0: 1, 1: 0}), cluster_assignments_kmeans)

            # Append values to lists for each iteration
            f_score_list_kmeans_iter.append(f_score_kmeans)
            ari_list_kmeans_iter.append(ari_kmeans)
            purity_list_kmeans_iter.append(purity_kmeans)

            f_score_list_gmm_iter.append(f_score_gmm)
            ari_list_gmm_iter.append(ari_gmm)
            purity_list_gmm_iter.append(purity_gmm)

            # Modified for AE
            f_score_list_ae_iter.append(f_score_ae)
            ari_list_ae_iter.append(ari_ae)
            purity_list_ae_iter.append(purity_ae)

        # Append mean values to lists for each method
        f_score_list_kmeans.append(np.mean(f_score_list_kmeans_iter))
        ari_list_kmeans.append(np.mean(ari_list_kmeans_iter))
        purity_list_kmeans.append(np.mean(purity_list_kmeans_iter))

        f_score_list_gmm.append(np.mean(f_score_list_gmm_iter))
        ari_list_gmm.append(np.mean(ari_list_gmm_iter))
        purity_list_gmm.append(np.mean(purity_list_gmm_iter))

        # Modified for AE
        f_score_list_ae.append(np.mean(f_score_list_ae_iter))
        ari_list_ae.append(np.mean(ari_list_ae_iter))
        purity_list_ae.append(np.mean(purity_list_ae_iter))

        # Plot and save the F-score for each method
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, f_score_list_kmeans_iter, label='K-Means F-score', marker='o')
        plt.plot(iterations, f_score_list_gmm_iter, label='GMM F-score', marker='o')
        plt.plot(iterations, f_score_list_ae_iter, label='Autoencoder F-score', marker='o')  # Modified for AE
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('F-score', fontsize=14)
        plt.title('Combined F-score')
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(dossier_graphes_fichier, f"combined_fscore_{fichier.split('.')[0]}.png"))
        plt.close()

        # Plot and save the ARI for each method
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, ari_list_kmeans_iter, label='K-Means ARI', marker='o')
        plt.plot(iterations, ari_list_gmm_iter, label='GMM ARI', marker='o')
        plt.plot(iterations, ari_list_ae_iter, label='Autoencoder ARI', marker='o')  # Modified for AE
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('ARI', fontsize=14)
        plt.title('Combined ARI')
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(dossier_graphes_fichier, f"combined_ari_{fichier.split('.')[0]}.png"))
        plt.close()

        # Plot and save the purity for each method
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, purity_list_kmeans_iter, label='K-Means Purity', marker='o')
        plt.plot(iterations, purity_list_gmm_iter, label='GMM Purity', marker='o')
        plt.plot(iterations, purity_list_ae_iter, label='Autoencoder Purity', marker='o')  # Modified for AE
        plt.xlabel('Iteration', fontsize=14)
        plt.ylabel('Purity', fontsize=14)
        plt.title('Combined Purity')
        plt.legend()
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(dossier_graphes_fichier, f"combined_purity_{fichier.split('.')[0]}.png"))
        plt.close()

        # Create a DataFrame with 'iterations' as the 'Iteration' column
        results_df = pd.DataFrame({
            'Iteration': iterations,
            'K-Means F-score': f_score_list_kmeans_iter,
            'K-Means ARI': ari_list_kmeans_iter,
            'K-Means Purity': purity_list_kmeans_iter,
            'GMM F-score': f_score_list_gmm_iter,
            'GMM ARI': ari_list_gmm_iter,
            'GMM Purity': purity_list_gmm_iter,
            'Autoencoder F-score': f_score_list_ae_iter,  # Modified for AE
            'Autoencoder ARI': ari_list_ae_iter,  # Modified for AE
            'Autoencoder Purity': purity_list_ae_iter,  # Modified for AE
        })

        # Append iteration time to the results dataframe
        results_df['Iteration Time (s)'] = time_list_iter

        # Save the results to a CSV file
        csv_filename = os.path.join(dossier_graphes_fichier, f"metrics_results_{fichier.split('.')[0]}.csv")

        if not os.path.exists(csv_filename):
            results_df.to_csv(csv_filename, index=False)
        else:
            results_df.to_csv(csv_filename, index=False, mode='a', header=False)
