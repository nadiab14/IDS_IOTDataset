import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim * 2)  # Two sets of parameters for mean and variance
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Sigmoid activation for reconstruction
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        latent_params = self.encoder(x)
        mu, log_var = latent_params[:, :self.latent_dim], latent_params[:, self.latent_dim:]
        z = self.reparameterize(mu, log_var)

        # Decode
        reconstructed = self.decoder(z)

        return reconstructed, mu, log_var

    def fit(self, X_train, X_test=None, epochs=100, batch_size=32, learning_rate=1e-3):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Convert data to PyTorch tensors
        train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        train_dataset = TensorDataset(train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_data = batch[0]
                reconstructed, mu, log_var = self(input_data)

                # Compute reconstruction loss and KL divergence
                reconstruction_loss = nn.functional.binary_cross_entropy(reconstructed, input_data, reduction='sum')
                kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                # Total loss
                loss = reconstruction_loss + kl_divergence
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss / len(train_loader.dataset)}")

    def transform(self, X):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Convert data to PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        # Encode only
        with torch.no_grad():
            latent_params = self.encoder(X_tensor)
            mu, _ = latent_params[:, :self.latent_dim], latent_params[:, self.latent_dim:]
            return mu.cpu().numpy()
