import torch
import torch.optim
import matplotlib.pyplot as plt
from collections import defaultdict
import utils
from AutoDecoder import AutoDecoder


def get_classwise_sample_indices(dataset, num_samples_per_class=2):
    """
    Get sample indices for constant sampling, where we pick `num_samples_per_class` 
    images from each class in the dataset.

    :param dataset: The dataset to sample from (assumes dataset returns (index, image, label))
    :param num_samples_per_class: Number of samples to retrieve per class
    :return: List of indices corresponding to selected samples
    """
    # Dictionary to hold class-wise indices
    class_indices = defaultdict(list)

    # Iterate through the dataset and collect indices for each class
    for idx, (_, _, label) in enumerate(dataset):
        class_indices[label].append(idx)

    # Consistently select `num_samples_per_class` from each class for comparison
    selected_indices = []
    for class_label, indices in class_indices.items():
        selected_indices.extend(indices[:num_samples_per_class])  # Take the first `num_samples_per_class` indices

    return selected_indices


def show_original_vs_reconstructed(model, latents, dataset, indices, num_samples=5, output_dir=None, filename="reconstructed_images.png"):
    """
    Save original vs reconstructed images side by side.

    :param model: Trained AutoDecoder model
    :param latents: Latent vectors of the dataset
    :param dataset: The dataset from which to sample images
    :param indices: Indices of the samples to display
    :param num_samples: Number of samples to display (default: 5)
    :param output_dir: Directory to save the output images
    :param filename: Filename to save the reconstructed images
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

    # Save the figure
    if output_dir:
        plt.savefig(f'{output_dir}/{filename}')
    else:
        plt.savefig(f'{filename}')
    plt.close()  # Close the figure to avoid displaying


def evaluate_model(model, data_loader, latents, device, hyperparameters, is_train_set=True, visualize=False, constant_sampling=True):
    """
    Evaluate the model on a given dataset. For the training set, it uses the passed latents.
    For the test set, it initializes and optimizes new latent vectors from a normal distribution.
    The hyperparameters are extracted from the passed dictionary.

    :param model: Trained AutoDecoder model
    :param data_loader: DataLoader for the dataset to evaluate (train_dl or test_dl)
    :param latents: Latent vectors for the training set or None for the test set
    :param device: Device to run the evaluation on (CPU or GPU)
    :param hyperparameters: Dictionary containing hyperparameters
    :param is_train_set: Boolean indicating if evaluating on the training set or test set
    :param visualize: Whether to visualize reconstructed images
    :param constant_sampling: Whether to sample two images per class
    :return: Average loss over the dataset
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    num_batches = len(data_loader)

    # Extract relevant hyperparameters
    latent_dim = hyperparameters['latent_dim']
    reconstruction_loss = hyperparameters['reconstruction_loss']
    epochs = hyperparameters.get('latent_epochs_for_test', 20)
    test_latent_lr = hyperparameters.get('latent_lr_for_test', 1e-4)
    normal_variance = hyperparameters.get('normal_latent_initialization_variance', 0.1)

    if is_train_set:
        # Evaluate on the training set using the stored latents
        with torch.no_grad():
            for _, (indices, x) in enumerate(data_loader):
                indices = indices.to(device)
                x = x.to(device).float() / 255.0  # Normalize the input to [0, 1]

                latent_vectors = latents[indices]  # Use stored latents for the training set
                x_rec = model(latent_vectors)

                # Compute the reconstruction loss
                loss = reconstruction_loss(x, x_rec)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches

        # Optionally visualize reconstructed images from the training set
        if visualize:
            if constant_sampling:
                sample_indices = get_classwise_sample_indices(data_loader.dataset, num_samples_per_class=2)  # Sample two per class
            else:
                sample_indices = torch.randint(0, len(data_loader.dataset), (5,))
            show_original_vs_reconstructed(model, latents, data_loader.dataset, sample_indices)

        return avg_loss
    else:
        # Initialize new latent vectors for the test set (from a normal distribution using the variance from the hyperparameters)
        test_latents = torch.normal(0, normal_variance, size=(len(data_loader.dataset), latent_dim), requires_grad=True, device=device)

        # Optimizer to optimize the test set latent vectors
        optimizer_latents = torch.optim.Adam([test_latents], lr=test_latent_lr)

        # Optimize latent vectors for the test set over a few epochs
        for epoch in range(epochs):
            epoch_loss = 0

            for _, (indices, x) in enumerate(data_loader):
                indices = indices.to(device)
                x = x.to(device).float() / 255.0  # Normalize the input to [0, 1]

                latent_vectors = test_latents[indices]  # Get latent vectors for the current batch
                x_rec = model(latent_vectors)  # Generate reconstructed images

                # Compute the reconstruction loss
                loss = reconstruction_loss(x, x_rec)

                optimizer_latents.zero_grad()  # Zero the gradients for the latent vectors
                loss.backward()
                optimizer_latents.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / num_batches
            print(f'Epoch [{epoch+1}/{epochs}], Test Set Loss: {avg_epoch_loss:.4f}')

        avg_loss = avg_epoch_loss

        # Optionally visualize reconstructed images from the test set
        if visualize:
            if constant_sampling:
                sample_indices = get_classwise_sample_indices(data_loader.dataset, num_samples_per_class=2)  # Sample two per class
            else:
                sample_indices = torch.randint(0, len(data_loader.dataset), (5,))
            show_original_vs_reconstructed(model, test_latents, data_loader.dataset, sample_indices)

        return avg_loss


def load_and_evaluate_model(model_path, latent_path, hyperparameters, output_dir):
    """
    Load the trained model and latent vectors from saved .pth files and evaluate on the training and test sets.
    The function extracts relevant hyperparameters from the passed dictionary.

    :param model_path: Path to the saved model (.pth) file
    :param latent_path: Path to the saved latent vectors (.pth) file
    :param hyperparameters: Dictionary containing hyperparameters
    :param output_dir: Directory to save the results
    :return: None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    train_ds, train_dl, test_ds, test_dl = utils.create_dataloaders(data_path="dataset", batch_size=hyperparameters['batch_size'])

    # Initialize the model with the same architecture
    model = AutoDecoder(latent_dim=hyperparameters['latent_dim'], 
                        feature_map_size=hyperparameters['feature_map_size'],
                        dropout_rate=hyperparameters['dropout_rate']).to(device)

    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Load the saved latent vectors for the training set
    latents = torch.load(latent_path, map_location=device)
    latents.requires_grad = False  # No need to optimize these again for training set evaluation

    # Evaluate on training set (using the saved latents)
    print("Evaluating on Training Set...")
    train_loss = evaluate_model(
        model=model, 
        data_loader=train_dl, 
        latents=latents, 
        device=device, 
        hyperparameters=hyperparameters, 
        output_dir=output_dir, 
        is_train_set=True, 
        visualize=True
    )
    print(f"Training Set Loss: {train_loss:.4f}")

    # Evaluate on test set (initializing and optimizing new latents)
    print("Evaluating on Test Set...")
    test_loss = evaluate_model(
        model=model, 
        data_loader=test_dl, 
        latents=None, 
        device=device, 
        hyperparameters=hyperparameters, 
        output_dir=output_dir, 
        is_train_set=False, 
        visualize=True
    )
    print(f"Test Set Loss: {test_loss:.4f}")
