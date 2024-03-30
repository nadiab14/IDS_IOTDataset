
import warnings

import pymc as pm
import numpy as np
import warnings

class VariationalBGMM:
    def __init__(self, q, num_clusters, alpha0=1.0, beta0=1.0, mu0=None, W0=None):
        self.q = q
        self.num_clusters = num_clusters
        self.alpha0 = alpha0
        self.beta0 = beta0

        if mu0 is None:
            mu0 = np.zeros(q.shape[1])  # Prior mean for the mean
        self.mu0 = mu0

        if W0 is None:
            W0 = np.eye(q.shape[1])  # Scale matrix for the covariance matrix
        self.W0 = W0
        self.nu0 = q.shape[1]  # Degrees of freedom for the covariance matrix

        self.q_z = np.random.dirichlet(np.ones(num_clusters), size=q.shape[0])

        # Initialize alpha with random values from the Dirichlet distribution
        self.alpha = np.random.dirichlet(alpha0 * np.ones(num_clusters))

        self.mu_k = None  # Will be initialized in the `fit` method

    def mv_normal_logp(self, value, mu, Sigma):
        n = mu.shape[0]
        try:
            Sigma_inv = np.linalg.inv(Sigma)
            det = np.linalg.det(Sigma)
            if det <= 0:
                det = np.finfo(float).eps  # Utilisation d'une valeur epsilon pour éviter les logarithmes invalides
        except np.linalg.LinAlgError:
            Sigma_reg = Sigma + 1e-6 * np.eye(Sigma.shape[0])
            Sigma_inv = np.linalg.inv(Sigma_reg)
            det = np.linalg.det(Sigma_reg)
            warnings.warn("La matrice Sigma est singulière. Régularisation appliquée.")

        diff = value - mu
        logp = -0.5 * n * np.log(2 * np.pi) - 0.5 * np.log(det) - 0.5 * np.sum(diff.dot(Sigma_inv) * diff, axis=1)
        return logp

    def update_assignments(self):
        for _ in range(self.num_iterations):
            alpha_new = self.alpha0 + np.sum(self.q_z, axis=0)

            for k in range(self.num_clusters):
                N_k = np.sum(self.q_z[:, k])
                if N_k > 0:
                    x_k = self.q.T.dot(self.q_z[:, k] / N_k)
                    S_k = (self.q.T * (self.q_z[:, k] / N_k)).dot(self.q) - x_k[:, np.newaxis].dot(x_k[np.newaxis, :])

                    mu_new = (self.beta0 * self.mu0 + N_k * x_k) / (self.beta0 + N_k)
                    self.q_z[:, k] = np.log(alpha_new[k]) + self.mv_normal_logp(self.q, mu_new, S_k)

            # Correction pour éviter les valeurs infinies ou NaN
            self.q_z = np.exp(self.q_z - np.max(self.q_z, axis=1, keepdims=True))
            self.q_z /= np.sum(self.q_z, axis=1, keepdims=True)

    def fit(self, num_iterations=1000):
        self.num_iterations = num_iterations

        # Create the PyMC3 model here and set mu_k and alpha as shared variables
        with pm.Model() as model:
            self.mu_k = pm.Data('mu_k_data', self.mu0, mutable=True)

            # Use Dirichlet distribution for the mixing coefficients (alpha)
            alpha = pm.Dirichlet('alpha', a=self.alpha)

            self.update_assignments()

    def get_assignments(self):
        return self.q_z

