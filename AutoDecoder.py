import torch.nn as nn
import argparse
import training
import evaluation_utils
import saving_utilities
from hyperparameters import auto_decoder_hyperparameters


class AutoDecoder(nn.Module):
    """
    AutoDecoder class that maps latent vectors to reconstructed images using 
    a fully connected layer and a decoder made of ConvTranspose2d layers.
    """

    def __init__(self, latent_dim=64, img_channels=1, feature_map_size=512, initial_map_size=7):
        """
        :param latent_dim: Dimensionality of the latent space
        :param img_channels: Number of image channels (1 for grayscale images in Fashion MNIST)
        :param feature_map_size: Number of feature maps to first extract from the latent vector, before going through CNN (e.g., 128/256/512)
        :param initial_map_size: Size of the initial spatial map before upsampling (e.g., 7 for a 7x7 map)
        """
        super().__init__()

        self.feature_map_size = feature_map_size
        self.initial_map_size = initial_map_size
        self.fc_output_size = self.initial_map_size * self.initial_map_size * self.feature_map_size

        # Fully connected layers to expand the latent vector into a feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.fc_output_size),
            nn.ReLU(),
            nn.Linear(self.fc_output_size, self.fc_output_size),
            nn.ReLU()
        )
        
        # Decoder architecture using ConvTranspose2d layers to upsample and reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_map_size, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),  # BatchNorm after ConvTranspose2d
            nn.ReLU(),
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),  # BatchNorm after ConvTranspose2d
            nn.ReLU(),
            
            nn.Conv2d(128, img_channels, kernel_size=3, padding=1),  # Output layer (28x28, 1 channel)
            nn.Sigmoid()  # Ensure output is between [0, 1]
        )

    def forward(self, z):
        """
        Forward pass of the auto-decoder.

        :param z: the latent vector for the sample
        :return: the reconstructed image
        """
        # Expand latent vector to feature map
        z = self.fc(z)
        # Reshape to (batch_size, feature_map_size, initial_map_size, initial_map_size)
        z = z.view(-1, self.feature_map_size, self.initial_map_size, self.initial_map_size)
        # Pass through decoder to upsample and reconstruct the image
        z = self.decoder(z)

        return z


def main(train_flag, results_dir=None):
    """
    Main function to either train and evaluate the model, or only evaluate the model based on the flag.
    
    :param train_flag: Boolean, if True the model will be trained, otherwise it will only evaluate from saved files.
    :param results_dir: The specific 'resultsX' directory to load the model from for evaluation. If None, it will load from the latest directory.
    """
    results_dir = None

    if train_flag:
        # Get the next results directory (resultsX) for training
        results_dir = saving_utilities.get_next_results_dir()
        # Save hyperparameters to file in the output_dir
        saving_utilities.save_hyperparameters(auto_decoder_hyperparameters, output_dir)
        # Train the model and save pth files in the output_dir
        training.train_auto_encoder(output_dir=output_dir, **auto_decoder_hyperparameters)
    
    # If no results directory is given, use the latest one
    results_dir = results_dir or saving_utilities.get_latest_results_dir()
    if results_dir is None:
        print("No valid results directory found. Cannot evaluate.")
        return

    # Evaluate the model using saved files from the results directory
    evaluation_utils.load_and_evaluate_model(
        model_path=f'{results_dir}/auto_decoder.pth',
        latent_path=f'{results_dir}/latent_vectors.pth',
        hyperparameters=auto_decoder_hyperparameters,
        output_dir=results_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the AutoDecoder model.")
    parser.add_argument("--train", action="store_true", help="Set this flag to train the model.")
    parser.add_argument("--results_dir", type=str, default=None, help="Specify the results directory (e.g., 'resultsX') to load for evaluation.")
    args = parser.parse_args()
    main(train_flag=args.train, results_dir=args.results_dir)

