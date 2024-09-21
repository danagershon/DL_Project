import torch
import torch.nn as nn

class VariationalAutoDecoder(nn.Module):
    """
    Variational AutoDecoder that maps latent vectors sampled from a distribution to reconstructed images,
    using a ConvTranspose2d decoder.
    """

    def __init__(self, latent_dim=64, img_channels=1, feature_map_size=512, initial_map_size=7, dropout_rate=0):
        """
        :param latent_dim: Dimensionality of the latent space
        :param img_channels: Number of image channels (1 for grayscale images in Fashion MNIST)
        :param feature_map_size: Number of feature maps to first extract from the latent vector, before going through CNN (e.g., 128/256/512)
        :param initial_map_size: Size of the initial spatial map before upsampling (e.g., 7 for a 7x7 map)
        :param dropout_rate: Dropout rate, if 0, no dropout will be applied
        """
        super().__init__()
        
        self.feature_map_size = feature_map_size
        self.initial_map_size = initial_map_size
        self.fc_output_size = self.initial_map_size * self.initial_map_size * self.feature_map_size
        self.dropout_rate = dropout_rate

        # Fully connected layers to extract a feature map from the latent vector
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, self.fc_output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(self.fc_output_size, self.fc_output_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        # Decoder architecture using ConvTranspose2d layers to upsample and reconstruct the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.feature_map_size, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(128, img_channels, kernel_size=3, padding=1),  # Output layer (28x28, 1 channel)
            nn.Sigmoid()  # Ensure output is between [0, 1]
        )

    def forward(self, z):
        """
        Forward pass of the variational auto-decoder.
        
        :param z: the pre-sampled latent vector
        :return: the reconstructed image
        """
        # Expand latent vector to feature map
        z = self.fc(z)
        # Reshape to (batch_size, feature_map_size, initial_map_size, initial_map_size)
        z = z.view(-1, self.feature_map_size, self.initial_map_size, self.initial_map_size)
        # Decode the latent vector into the reconstructed image
        z = self.decoder(z)

        return z
