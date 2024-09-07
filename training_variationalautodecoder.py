import time
import torch
import matplotlib.pyplot as plt
import utils
import losses
import saving_utilities
from VariationalAutoDecoder import VariationalAutoDecoder


def train_variational_auto_decoder(batch_size=32, 
                       latent_dim=256, 
                       feature_map_size=256, 
                       epochs=100, 
                       lr=1e-4, 
                       reconstruction_loss=losses.reconstruction_loss_BCE, #TODO LEFT Replace with ELBO 
                       latent_initialization="normal", 
                       latent_reg_loss_lambda=1e-5, 
                       normal_latent_initialization_variance=0.1,
                       patience=10,
                       dropout_rate=0,
                       output_dir=None,
                       **kw):
    """
    Train the VAD on the Fashion MNIST dataset with Early Stopping.

    :param batch_size: Size of the mini-batches used for training
    :param latent_dim: Dimensionality of the latent space
    :param feature_map_size: Number of feature maps to first extract from the latent vector, before going through CNN
    :param epochs: Number of training epochs
    :param lr: Learning rate for the optimizer
    :param reconstruction_loss: Loss function for reconstruction (BCE or MSE)
    :param latent_initialization: the distribution type to initialize latent vectors from ('normal', 'random', 'uniform')
    :param latent_reg_loss_lambda: L2 regularization strength on latent vectors (if 0, there is no regularization)
    :param patience: Number of epochs with no improvement after which training will be stopped
    :param dropout_rate: dropout rate for the Auto Decoder architecture
    :param output_dir: Directory to save the output files
    """
    # load fashion-MNIST dataset
    train_ds, train_dl, test_ds, test_dl = utils.create_dataloaders(data_path="dataset", batch_size=batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    model = VariationalAutoDecoder(latent_dim=latent_dim, 
                        feature_map_size=feature_map_size, 
                        dropout_rate=dropout_rate).to(device)
    
    # Save the model architecture into a file
    saving_utilities.save_model_architecture(model, output_dir)

    # Initialize a random latent vector for each sample in the training set
    if latent_initialization == "random":
        latents = torch.randn(len(train_ds), latent_dim, requires_grad=True, device=device)
    elif latent_initialization == "uniform":
        latents = torch.rand(len(train_ds), latent_dim, requires_grad=True, device=device)
    else:  # normal
        latents = torch.normal(0, normal_latent_initialization_variance, size=(len(train_ds), latent_dim), requires_grad=True, device=device)

    # define optimizers
    optimizer_model = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_latents = torch.optim.Adam([latents], lr=lr)

    epoch_losses = []
    best_loss = float('inf')
    early_stop_counter = 0

    # Start total training time measurement
    total_start_time = time.time()

    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()  # Start measuring time for this epoch
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

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / len(train_dl)
        epoch_losses.append(avg_epoch_loss)

        # Early stopping condition: check if loss improves
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            early_stop_counter = 0  # Reset the counter if there is an improvement
        else:
            early_stop_counter += 1

        # If early stop counter reaches the patience limit, stop training
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}, best loss: {best_loss:.4f}")
            break

        # Calculate time taken for the epoch
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = epoch_end_time - total_start_time

        # Print the average loss, time for this epoch, and total elapsed time
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_epoch_loss:.4f}, Epoch Time: {epoch_duration:.2f} seconds, Total Time: {total_elapsed_time:.2f} seconds')

    # End total training time measurement
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    # Print total training time
    print(f'Total training time: {total_training_time/60:.2f} min')

    # Save model and latent vectors after training
    torch.save(model.state_dict(), f'{output_dir}/auto_decoder.pth')
    torch.save(latents, f'{output_dir}/latent_vectors.pth')

    # ---------------- Save the Loss Plot ----------------
    plt.figure()
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/loss_plot.png')  # Save the loss plot to a file
    plt.close()  # Close the plot to avoid displaying