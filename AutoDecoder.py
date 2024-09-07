import torch.nn as nn


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


if __name__ == "__main__":
    import training
    from hyperparameters import auto_decoder_hyperparameters
    import evaluation_utils

    train_model = False
    if train_model:
        training.train_auto_encoder(**auto_decoder_hyperparameters)

    evaluation_utils.load_and_evaluate_model(model_path='auto_decoder.pth', 
                                             latent_path='latent_vectors.pth',
                                             hyperparameters=auto_decoder_hyperparameters)

