import torch
import torch.nn as nn

import torch.optim
import matplotlib.pyplot as plt

import utils


class AutoDecoder(nn.Module):
    def __init__(self, latent_dim=64, img_channels=1):
        """
        Initialize the AutoDecoder.
        :param latent_dim: Dimensionality of the latent space
        :param img_channels: Number of image channels (1 for grayscale images in Fashion MNIST)
        """
        super().__init__()

        # Fully connected layers to expand the latent vector into a feature map
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)  # Increase feature map size to 128 channels
        
        # Decoder architecture using ConvTranspose2d layers to reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.ReLU(),
            nn.Conv2d(32, img_channels, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        Forward pass of the auto-decoder.
        :param z: the latent vector for each sample
        :return: the reconstructed image
        """
        z = self.fc(z)
        z = z.view(-1, 128, 7, 7)  # Reshape to (batch_size, channels, height, width)
        z = self.decoder(z)  # Apply the CNN decoder to reconstruct the image
        return z
    

def reconstruction_loss_MSE(x, x_rec):
    return torch.mean((x - x_rec) ** 2)


def reconstruction_loss_CE(x, x_rec):
    return torch.nn.functional.binary_cross_entropy(x_rec, x)


def train_auto_encoder(batch_size=32, latent_dim=64, epochs=100, lr=1e-3, reconstruction_loss=reconstruction_loss_MSE,
                       latent_initialization="normal", add_reg_loss=False):
    train_ds, train_dl, test_ds, test_dl = utils.create_dataloaders(data_path="dataset", batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    model = AutoDecoder(latent_dim=latent_dim).to(device)

    # Initialize latent vectors for each sample in the training set
    if latent_initialization == "random":
        latents = torch.randn(len(train_ds), latent_dim, requires_grad=True, device=device)  # Random initialization of latents
    elif latent_initialization == "uniform":
        latents = torch.rand(len(train_ds), latent_dim, requires_grad=True, device=device)
    else:  # normal
        latents = torch.normal(0, 0.01, size=(len(train_ds), latent_dim), requires_grad=True, device=device)

    # Optimizers
    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_latents = torch.optim.Adam([latents], lr=lr)  # Optimizing the latents as well

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for _, (indices, x) in enumerate(train_dl):
            # Move data to the device (GPU/CPU)
            indices = indices.to(device)
            x = x.to(device).float() / 255.0  # Normalize the input to [0, 1]

            # Forward pass: get reconstructed images
            latent_vectors = latents[indices]
            x_rec = model(latent_vectors)

            # Compute the reconstruction loss
            if add_reg_loss:
                reg_loss = 1e-4 * torch.norm(latents, p=2)  # Define the regularization term (L2 regularization)
                loss = reconstruction_loss(x, x_rec) + reg_loss
            else:
                loss = reconstruction_loss(x, x_rec)

            # Backpropagation
            optimizer_model.zero_grad()
            optimizer_latents.zero_grad()
            loss.backward()

            # Update the model and latent vectors
            optimizer_model.step()
            optimizer_latents.step()

            # Accumulate the loss for the epoch
            epoch_loss += loss.item()

        # Print the average loss per epoch
        avg_epoch_loss = epoch_loss / len(train_dl)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}')

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