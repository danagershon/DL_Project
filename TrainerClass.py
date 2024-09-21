import torch
import matplotlib.pyplot as plt
import utils
import losses
import saving_utilities
from AutoDecoder import AutoDecoder
from VariationalAutoDecoderNew import VariationalAutoDecoder


class BaseTrainer:

    def __init__(self, 
                 output_dir, 
                 model_filename, 
                 latent_filename,
                 batch_size=32, 
                 latent_dim=256, 
                 feature_map_size=256, 
                 epochs=100, 
                 lr=1e-4, 
                 reconstruction_loss=losses.reconstruction_loss_BCE, 
                 latent_initialization="normal", 
                 latent_reg_loss_lambda=1e-5, 
                 normal_latent_initialization_variance=0.1, 
                 patience=10, 
                 dropout_rate=0,
                 **kw
                 ):
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.feature_map_size = feature_map_size
        self.epochs = epochs
        self.lr = lr
        self.reconstruction_loss = reconstruction_loss
        self.latent_reg_loss_lambda = latent_reg_loss_lambda
        self.patience = patience
        self.dropout_rate = dropout_rate
        self.output_dir = output_dir
        self.latent_initialization = latent_initialization  # normal, random, uniform (for AD)
        self.normal_latent_initialization_variance = normal_latent_initialization_variance  # for AD
        self.model_filename = model_filename  # auto_decoder.pth for AD, variational_auto_decoder.pth for VAD
        self.latent_filename = latent_filename  # latent_vectors.pth for AD, latent_params.pth for VAD

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.train_dl = None
        self.train_ds = None
        self.latent_params = None  # [latents] for AD, [mu, logvar] for VAD

    def initialize_model(self):
        """This method should be overridden by child classes to initialize the model."""
        raise NotImplementedError

    def initialize_latents(self):
        """This method should be overridden by child classes to initialize latent (AD) vectors or latent parameters (VAD)."""
        raise NotImplementedError

    def get_latent_vectors(self, indices):
        """This method should be overridden by child classes to retrieve latent vectors."""
        raise NotImplementedError

    def compute_loss(self, x, x_rec, indices):
        """This method should be overridden by child classes to compute the total loss (reconstruction + regularization)."""
        raise NotImplementedError

    def train(self):
        # Load Fashion-MNIST dataset
        self.train_ds, self.train_dl, _, _ = utils.create_dataloaders(data_path="dataset", batch_size=self.batch_size)
        self.initialize_model()
        self.initialize_latents()  # latents for AD, (mu, logvar) for VAD

        optimizer_model = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        optimizer_latents = torch.optim.Adam(self.latent_params, lr=self.lr)

        epoch_losses = []
        best_loss = float('inf')
        early_stop_counter = 0

        # Training loop
        for epoch in range(self.epochs):
            self.model.train()  # set to train mode
            epoch_loss = 0

            for _, (indices, x) in enumerate(self.train_dl):
                indices = indices.to(self.device)
                x = x.to(self.device).float() / 255.0  # Normalize the input to [0, 1]

                # Get latent vectors (for VAD they are sampled from distributions) 
                z = self.get_latent_vectors(indices)

                # Forward pass: generate reconstructed images
                x_rec = self.model(z)

                # Compute the loss (reconstruction + reg for AD, ELBO for VAD)
                loss = self.compute_loss(x, x_rec, indices)

                # Backpropagation
                optimizer_model.zero_grad()
                optimizer_latents.zero_grad()
                loss.backward()

                # Update parameters
                optimizer_model.step()
                optimizer_latents.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_dl)
            epoch_losses.append(avg_epoch_loss)

            # Early stopping condition
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss:.4f}")
                break

            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_epoch_loss:.4f}')

        # Save model and latents after training
        self.save_results(epoch_losses)

    def save_results(self, epoch_losses):
        # Save model and latent vectors
        torch.save(self.model.state_dict(), f'{self.output_dir}/{self.model_filename}')
        torch.save(self.latent_params, f'{self.output_dir}/{self.latent_filename}')

        # Plot loss
        self.plot_loss(epoch_losses)

    def plot_loss(self, epoch_losses):
        # save loss graph plot to file
        plt.figure()
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.output_dir}/loss_plot.png')
        plt.close()


class ADTrainer(BaseTrainer):

    def initialize_model(self):
        self.model = AutoDecoder(latent_dim=self.latent_dim, feature_map_size=self.feature_map_size, dropout_rate=self.dropout_rate).to(self.device)
        saving_utilities.save_model_architecture(self.model, self.output_dir)

    def initialize_latents(self):
        if self.latent_initialization == "random":
            latent_vectors = torch.randn(len(self.train_ds), self.latent_dim, requires_grad=True, device=self.device)
        elif self.latent_initialization == "uniform":
            latent_vectors = torch.rand(len(self.train_ds), self.latent_dim, requires_grad=True, device=self.device)
        else:  # normal
            latent_vectors = torch.normal(0, self.normal_latent_initialization_variance, size=(len(self.train_ds), self.latent_dim), requires_grad=True, device=self.device)

        self.latent_params = [latent_vectors]

    def get_latent_vectors(self, indices):
        latent_vectors = self.latent_params[0]  # latent_params has just the latents
        return latent_vectors[indices]

    def compute_loss(self, x, x_rec, indices):
        recon_loss = self.reconstruction_loss(x, x_rec)
        # L2 regularization loss for the latent vectors
        latent_vectors = self.latent_params[0]
        batch_latent_vectors = latent_vectors[indices]
        reg_loss = self.latent_reg_loss_lambda * torch.norm(batch_latent_vectors, p=2)

        return recon_loss + reg_loss


class VADTrainer(BaseTrainer):

    def __init__(self, output_dir, 
                 model_filename, 
                 latent_filename, 
                 batch_size=32, 
                 latent_dim=256, 
                 feature_map_size=256, 
                 epochs=100, lr=0.0001, 
                 reconstruction_loss=losses.reconstruction_loss_BCE, 
                 latent_initialization="normal", 
                 latent_reg_loss_lambda=0.00001, 
                 normal_latent_initialization_variance=0.1, 
                 patience=10, 
                 dropout_rate=0, 
                 kl_weight=0.001,  # exclusive for VAD
                 **kw
                 ):
        super().__init__(output_dir, model_filename, latent_filename, batch_size, latent_dim, feature_map_size, epochs, lr, reconstruction_loss, latent_initialization, latent_reg_loss_lambda, normal_latent_initialization_variance, patience, dropout_rate, **kw)
        self.kl_weight = kl_weight

    def initialize_model(self):
        self.model = VariationalAutoDecoder(latent_dim=self.latent_dim, feature_map_size=self.feature_map_size, dropout_rate=self.dropout_rate).to(self.device)
        saving_utilities.save_model_architecture(self.model, self.output_dir)

    def initialize_latents(self):
        # initialize mu from normal distribution
        mu = torch.normal(0, self.normal_latent_initialization_variance, size=(len(self.train_ds), self.latent_dim), requires_grad=True, device=self.device)
        # initialize log variance (log(sigma^2)) close to -1 for stability
        logvar = torch.full_like(mu, -1.0, requires_grad=True) 
        self.latent_params = [mu, logvar]

    def get_latent_vectors(self, indices):
        mu, logvar = self.latent_params
        batch_mu = mu[indices]
        batch_logvar = logvar[indices]
        sigma = torch.exp(0.5 * batch_logvar)  # logvar is log(sigma^2)
        eps = torch.randn_like(sigma)  # sample from N(0,1)
        z = batch_mu + sigma * eps   # Reparameterization trick

        return z

    def compute_loss(self, x, x_rec, indices):
        # compute ELBO loss
        mu, logvar = self.latent_params
        batch_mu = mu[indices]
        batch_logvar = logvar[indices]

        recon_loss = self.reconstruction_loss(x, x_rec)

        # KL Divergence loss (weighted)
        kl_loss = -0.5 * torch.sum(1 + batch_logvar - batch_mu.pow(2) - batch_logvar.exp()) / self.batch_size  # TODO: why divide by batch size?

        # L2 regularization loss for the latent vectors
        reg_loss = self.latent_reg_loss_lambda * torch.norm(mu, p=2)

        return recon_loss + self.kl_weight * kl_loss + reg_loss
