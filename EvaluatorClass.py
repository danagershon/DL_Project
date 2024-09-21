
import torch
import torch.optim
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import utils
from AutoDecoder import AutoDecoder
from VariationalAutoDecoderNew import VariationalAutoDecoder


class EvaluatorBase:

    def __init__(self, hyperparameters, model_filename, latent_filename, output_dir, visualize=True, constant_sampling=True):
        self.hyperparameters = hyperparameters
        self.model_path = os.path.join(output_dir, model_filename)
        self.latent_path = os.path.join(output_dir, latent_filename)
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.visualize = visualize
        self.constant_sampling = constant_sampling

        self.model_cls = None  # to be determined by child class

    def load_model_from_pth(self):
        # initialize model
        model = self.model_cls(
            latent_dim=self.hyperparameters['latent_dim'],
            feature_map_size=self.hyperparameters['feature_map_size'],
            dropout_rate=self.hyperparameters['dropout_rate']
        ).to(self.device)

        # load model from .pth file
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        return model

    def load_latents(self):
        raise NotImplementedError
    
    def compute_loss(self, x, x_rec, **kw):
        raise NotImplementedError

    def evaluate_train(self, data_loader):
        raise NotImplementedError

    def evaluate_test(self, data_loader):
        raise NotImplementedError
    
    def evaluate(self):
        # Load the dataset
        _, train_dl, _, test_dl = utils.create_dataloaders(data_path="dataset", batch_size=self.hyperparameters['batch_size'])

        # Evaluate on training set
        print("Evaluating on Training Set...")
        train_loss = self.evaluate_train(train_dl)
        print(f"Training Set Loss: {train_loss:.4f}")

        # Evaluate on test set
        print("Evaluating on Test Set...")
        test_loss = self.evaluate_test(test_dl)
        print(f"Test Set Loss: {test_loss:.4f}")
    
    def get_classwise_sample_indices(self, dataset, num_samples_per_class=1):
        """
        Get sample indices for constant sampling, where we pick `num_samples_per_class` 
        images from each class in the dataset.

        :param dataset: The dataset to sample from (assumes dataset returns (image, label))
        :param num_samples_per_class: Number of samples to retrieve per class
        :return: List of indices corresponding to selected samples
        """
        # Dictionary to hold class-wise indices
        class_indices = defaultdict(list)

        # Iterate through the dataset and collect indices for each class
        for idx, (image, label) in enumerate(dataset):
            class_indices[label].append(idx)

        # Consistently select `num_samples_per_class` from each class for comparison
        selected_indices = []
        for class_label, indices in class_indices.items():
            selected_indices.extend(indices[:num_samples_per_class])  # Take the first `num_samples_per_class` indices

        return selected_indices
    
    def visualize_sampled_images_and_rec(self, model, latents, dataset, filename, is_VAD):
        if self.visualize:
            if self.constant_sampling:
                sample_indices = self.get_classwise_sample_indices(dataset)
            else:
                sample_indices = torch.randint(0, len(dataset), (5,))

            self.visualize_results(model, latents, dataset, sample_indices, filename, is_VAD)

    
    def visualize_results(self, model, latents, dataset, indices, filename, is_VAD):
        """
        Save original vs reconstructed images side by side.
        """
        num_samples = min(5, len(indices))  # currently limited to 5 regargless of num indices due to plot being too big 
        model.eval()  # set model to evaluation mode

        with torch.no_grad():
            original_images = []
            reconstructed_images = []

            for idx in indices[0:num_samples]:
                # Original image
                data_item = dataset[idx]  
                # Since dataset returns (index, image), we use data_item[1] to get the image
                original_img = data_item[1].float() / 255.0  # Normalize to [0, 1]
                original_images.append(original_img)

                # Reconstructed image
                latent_vector = latents[idx].unsqueeze(0) if not is_VAD else latents[idx]  # TODO: check of logic for VAD works
                reconstructed_img = model(latent_vector).squeeze(0)  # Remove batch dimension
                reconstructed_images.append(reconstructed_img)

        # Plot original vs reconstructed images in a grid (2 rows: 1 for original, 1 for reconstructed)
        fig, axs = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        for i in range(num_samples):
            # Original image
            axs[0, i].imshow(original_images[i].cpu().squeeze(), cmap='gray')  # Use .squeeze() to remove channel dimension
            axs[0, i].axis('off')
            axs[0, i].set_title('Original')

            # Reconstructed image
            axs[1, i].imshow(reconstructed_images[i].cpu().squeeze(), cmap='gray')  # Use .squeeze() to remove channel dimension
            axs[1, i].axis('off')
            axs[1, i].set_title('Reconstructed')

        plt.savefig(f'{self.output_dir}/{filename}')
        plt.close()


class EvaluatorAD(EvaluatorBase):

    def __init__(self, hyperparameters, model_filename, latent_filename, output_dir):
        super().__init__(hyperparameters, model_filename, latent_filename, output_dir)
        self.model_cls = AutoDecoder

    def load_latents(self):
        # load latent vectors from .pth file
        latent_params = torch.load(self.latent_path, map_location=self.device)
        latents = latent_params[0]
        latents.requires_grad = False

        return latents
    
    def compute_loss(self, x, x_rec):
        # copmute reconstruction loss w/o latent reg
        return self.hyperparameters['reconstruction_loss'](x, x_rec)

    def evaluate_train(self, data_loader):
        model = self.load_model_from_pth()
        latents = self.load_latents()

        model.eval()  # set model to evaluation mode
        total_loss = 0
        num_batches = len(data_loader)

        # Evaluate on the training set using the stored latents
        with torch.no_grad():
            for _, (indices, x) in enumerate(data_loader):
                indices = indices.to(self.device)
                x = x.to(self.device).float() / 255.0  # Normalize the input to [0, 1]

                latent_vectors = latents[indices]
                x_rec = model(latent_vectors)

                loss = self.compute_loss(x, x_rec)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches

        self.visualize_sampled_images_and_rec(model, latents, data_loader.dataset, 'ad_train_reconstructions.png', is_VAD=False)

        return avg_loss

    def evaluate_test(self, data_loader):
        model = self.load_model_from_pth()
        latent_dim = self.hyperparameters['latent_dim']
        test_latents = torch.normal(0, self.hyperparameters['normal_latent_initialization_variance'], size=(len(data_loader.dataset), latent_dim), requires_grad=True, device=self.device)

        optimizer_latents = torch.optim.Adam([test_latents], lr=self.hyperparameters['latent_lr_for_test'])
        model.eval()  # set model to evaluation mode
        total_loss = 0
        num_batches = len(data_loader)
        epochs = self.hyperparameters['latent_epochs_for_test']

        for epoch in range(epochs):
            epoch_loss = 0

            for _, (indices, x) in enumerate(data_loader):
                indices = indices.to(self.device)
                x = x.to(self.device).float() / 255.0  # Normalize the input to [0, 1]

                latent_vectors = test_latents[indices]
                x_rec = model(latent_vectors)

                loss = self.compute_loss(x, x_rec)

                optimizer_latents.zero_grad()
                loss.backward()
                optimizer_latents.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / num_batches
            print(f'Epoch [{epoch+1}/{epochs}], Test Set Loss: {avg_epoch_loss:.4f}')
            total_loss += avg_epoch_loss

        avg_loss = total_loss / self.hyperparameters['latent_epochs_for_test']

        self.visualize_sampled_images_and_rec(model, test_latents, data_loader.dataset, 'ad_test_reconstructions.png', is_VAD=False)

        return avg_loss


class EvaluatorVAD(EvaluatorBase):

    def __init__(self, hyperparameters, model_filename, latent_filename, output_dir):
        super().__init__(hyperparameters, model_filename, latent_filename, output_dir)
        self.model_cls = VariationalAutoDecoder

    def load_latents(self):
        # load mu and logvar from .pth file
        latent_params = torch.load(self.latent_path, map_location=self.device)
        mu = latent_params[0]
        logvar = latent_params[1]

        return mu, logvar
    
    def compute_loss(self, x, x_rec, batch_logvar, batch_mu):
        # compute ELBO loss w/o latent reg
        recon_loss = self.hyperparameters['reconstruction_loss'](x, x_rec)
        
        # KL Divergence loss (weighted)
        kl_loss = -0.5 * torch.sum(1 + batch_logvar - batch_mu.pow(2) - batch_logvar.exp()) / x.size(0)

        return recon_loss + self.hyperparameters['kl_weight'] * kl_loss

    def evaluate_train(self, data_loader):
        model = self.load_model_from_pth()
        mu, logvar = self.load_latents()

        model.eval()
        total_loss = 0
        num_batches = len(data_loader)

        with torch.no_grad():
            for _, (indices, x) in enumerate(data_loader):
                indices = indices.to(self.device)
                x = x.to(self.device).float() / 255.0  # Normalize the input to [0, 1]

                batch_mu = mu[indices]
                batch_logvar = logvar[indices]
                std = torch.exp(0.5 * batch_logvar)
                eps = torch.randn_like(std)
                z = batch_mu + eps * std

                x_rec = model(z)

                loss = self.compute_loss(x, x_rec, batch_logvar, batch_mu)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches

        # we pass the mu as latents becuase mu is the mean latent in the space, therfore a good representor
        self.visualize_sampled_images_and_rec(model, mu, data_loader.dataset, 'vad_train_reconstructions.png', is_VAD=True)

        return avg_loss

    def evaluate_test(self, data_loader):
        model = self.load_model_from_pth()
        latent_dim = self.hyperparameters['latent_dim']
        test_mu = torch.normal(0, self.hyperparameters['normal_latent_initialization_variance'], size=(len(data_loader.dataset), latent_dim), requires_grad=True, device=self.device)
        test_logvar = torch.full_like(test_mu, -1.0, requires_grad=True)

        optimizer_latents = torch.optim.Adam([test_mu, test_logvar], lr=self.hyperparameters['latent_lr_for_test'])
        model.eval()
        total_loss = 0
        num_batches = len(data_loader)
        epochs = self.hyperparameters['latent_epochs_for_test']

        for epoch in range(epochs):
            epoch_loss = 0

            for _, (indices, x) in enumerate(data_loader):
                indices = indices.to(self.device)
                x = x.to(self.device).float() / 255.0  # Normalize the input to [0, 1]

                batch_mu = test_mu[indices]
                batch_logvar = test_logvar[indices]
                std = torch.exp(0.5 * batch_logvar)
                eps = torch.randn_like(std)
                z = batch_mu + eps * std

                x_rec = model(z)

                loss = self.compute_loss(x, x_rec, batch_logvar, batch_mu)

                optimizer_latents.zero_grad()
                loss.backward()
                optimizer_latents.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / num_batches
            print(f'Epoch [{epoch+1}/{epochs}], Test Set Loss: {avg_epoch_loss:.4f}')
            total_loss += avg_epoch_loss

        avg_loss = total_loss / self.hyperparameters['latent_epochs_for_test']

        # we pass the mu as latents becuase mu is the mean latent in the space, therefore a good representative
        self.visualize_sampled_images_and_rec(model, test_mu, data_loader.dataset, 'vad_test_reconstructions.png', is_VAD=True)

        return avg_loss
