import torch
import torch.nn as nn
import torch.optim
import matplotlib.pyplot as plt
import time
import utils


class AutoDecoder(nn.Module):
    """
    AutoDecoder class that maps latent vectors to reconstructed images using 
    a fully connected layer and a decoder made of ConvTranspose2d layers.
    """

    def __init__(self, latent_dim=64, img_channels=1, feature_map_size=512):
        """
        :param latent_dim: Dimensionality of the latent space
        :param img_channels: Number of image channels (1 for grayscale images in Fashion MNIST)
        :param feature_map_size: Number of feature maps to first extract from the latent vector, before going throght CNN (e.g., 128/256/512)
        """
        super().__init__()

        # Expand the latent vector into a feature map via a fully connected layer
        self.feature_map_size = feature_map_size
        self.fc = nn.Linear(latent_dim, 7 * 7 * feature_map_size)
        
        # Decoder architecture. Using ConvTranspose2d layers to "reconstruct" the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(128, img_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Forward pass of the auto-decoder.

        :param z: the latent vector for the sample
        :return: the reconstructed image
        """
        # Extract feature map from the latent vector
        z = self.fc(z)
        # Reshape to (batch_size, feature_map_size, height, width)
        z = z.view(-1, self.feature_map_size, 7, 7)
        # Reconstruct the image with decoder
        z = self.decoder(z)

        return z


def reconstruction_loss_MSE(x, x_rec):
    """
    Compute the mean squared error (MSE) between original and reconstructed images.

    :param x: Original images
    :param x_rec: Reconstructed images
    :return: MSE loss
    """
    return torch.mean((x - x_rec) ** 2)


def reconstruction_loss_BCE(x, x_rec):
    """
    Compute the binary cross entropy (BCE) between original and reconstructed images.

    :param x: Original images
    :param x_rec: Reconstructed images
    :return: BCE loss
    """
    # Add a channel dimension to x so it matches x_rec's shape
    x = x.unsqueeze(1)  # Shape will become [batch_size, 1, 28, 28]
    return torch.nn.functional.binary_cross_entropy(x_rec, x)


def train_auto_encoder(batch_size=32, 
                       latent_dim=256, 
                       feature_map_size=256, 
                       epochs=100, 
                       lr=1e-4, 
                       reconstruction_loss=reconstruction_loss_BCE,
                       latent_initialization="normal", 
                       latent_reg_loss_lambda=1e-5, 
                       normal_latent_initialization_variance=0.1):
    """
    Train the AutoDecoder on the Fashion MNIST dataset.

    All of the function parameters are hyperparameters:

    :param batch_size: Size of the mini-batches used for training
    :param latent_dim: Dimensionality of the latent space
    :param feature_map_size: Number of feature maps to first extract from the latent vector, before going throght CNN (e.g., 128/256/512)
    :param epochs: Number of training epochs
    :param lr: Learning rate for the optimizer
    :param reconstruction_loss: Loss function for reconstruction (BCE or MSE)
    :param latent_initialization: the distribution type to initialize latent vectors from ('normal', 'random', 'uniform')
    :param latent_reg_loss_lambda: L2 regularization strength on latent vectors (if 0, there is no regularization)
    :param normal_latent_initialization_variance: Variance used for normal initialization of latent vectors
    """
    # load fashion-MNIST
    train_ds, train_dl, test_ds, test_dl = utils.create_dataloaders(data_path="dataset", batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    model = AutoDecoder(latent_dim=latent_dim, feature_map_size=feature_map_size).to(device)

    # Initialize a random latent vector for each sample in the training set
    if latent_initialization == "random":
        latents = torch.randn(len(train_ds), latent_dim, requires_grad=True, device=device)
    elif latent_initialization == "uniform":
        latents = torch.rand(len(train_ds), latent_dim, requires_grad=True, device=device)
    else:  # normal
        latents = torch.normal(0, normal_latent_initialization_variance, size=(len(train_ds), latent_dim), requires_grad=True, device=device)

    # define optimizers
    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
    # Since it ia an Auto Decoder, we optimize the latents as well
    optimizer_latents = torch.optim.Adam([latents], lr=lr)  
    # TODO: consider different lr for the model and latents
    # TODO: consider a different optimizer (though Adam is considered good in general)

    # Start total training time measurement
    total_start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()  # Start measuring time for this epoch
        model.train()  # set model to training mode
        epoch_loss = 0

        for _, (indices, x) in enumerate(train_dl):
            # Move data to the device (GPU/CPU)
            indices = indices.to(device)
            x = x.to(device).float() / 255.0  # Normalize the input to [0, 1]

            # Forward pass: generate reconstructed images from latents
            latent_vectors = latents[indices]
            x_rec = model(latent_vectors)

            # Compute the reconstruction loss
            loss = reconstruction_loss(x, x_rec) + latent_reg_loss_lambda * torch.norm(latents, p=2)

            # Backpropagation
            optimizer_model.zero_grad()
            optimizer_latents.zero_grad()
            loss.backward()

            # Update the model params and latent vectors
            optimizer_model.step()
            optimizer_latents.step()

            # Accumulate the loss for the epoch
            epoch_loss += loss.item()

        # Calculate time taken for the epoch
        end_time = time.time()
        epoch_duration = end_time - start_time

        # Print the average loss per epoch
        avg_epoch_loss = epoch_loss / len(train_dl)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

    # End total training time measurement
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    # Print total training time
    print(f'Total training time: {total_training_time:.2f} seconds')

    # Save model and latent vectors after training
    torch.save(model.state_dict(), 'auto_decoder.pth')
    torch.save(latents, 'latent_vectors.pth')

    # ---------------- visual evaluation ----------------
    sample_indices = torch.randint(0, len(train_ds), (5,))
    # Display original vs reconstructed images
    show_original_vs_reconstructed(model, latents, train_ds, sample_indices)


def show_original_vs_reconstructed(model, latents, dataset, indices, num_samples=5):
    """
    Display original vs reconstructed images side by side.

    :param model: Trained AutoDecoder model
    :param latents: Latent vectors of the dataset
    :param dataset: The dataset from which to sample images
    :param indices: Indices of the samples to display
    :param num_samples: Number of samples to display (default: 5)
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Get original and reconstructed images
        original_images = []
        reconstructed_images = []
        
        for idx in indices[:num_samples]:
            # Original image
            original_img = dataset[idx][1].float() / 255.0  # Normalize to [0, 1]
            original_images.append(original_img)
            
            # Reconstructed image
            latent_vector = latents[idx].unsqueeze(0)  # Add batch dimension
            reconstructed_img = model(latent_vector).squeeze(0)  # Remove batch dimension
            reconstructed_images.append(reconstructed_img)
    
    # Plot original vs reconstructed images
    fig, axs = plt.subplots(2, num_samples, figsize=(num_samples*2, 4))
    for i in range(num_samples):
        # Original image
        axs[0, i].imshow(original_images[i].cpu().squeeze(), cmap='gray')  # Use .squeeze() to remove channel dimension
        axs[0, i].axis('off')
        axs[0, i].set_title('Original')

        # Reconstructed image
        axs[1, i].imshow(reconstructed_images[i].cpu().squeeze(), cmap='gray')  # Use .squeeze() to remove channel dimension
        axs[1, i].axis('off')
        axs[1, i].set_title('Reconstructed')

    plt.show()


if __name__ == "__main__":
    train_auto_encoder()
