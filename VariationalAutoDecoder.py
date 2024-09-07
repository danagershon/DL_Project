import torch
import torch.nn as nn

class VariationalAutoDecoder(nn.Module):
    """
    Variational AutoDecoder class that maps latent vectors to distribution parameters, to generate new latent vectors and reconstruct images using 
    a fully connected layer and a decoder made of ConvTranspose2d layers with optional dropout.
    """

    def __init__(self, input_dim=64, latent_dim=64, img_channels=1, feature_map_size=512, initial_map_size=7, dropout_rate=0):
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
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # Fully connected layers to expand the latent vector into a feature map
        fc_layers = [
            nn.Linear(input_dim, self.fc_output_size),
            nn.ReLU()
        ]

        if dropout_rate > 0:
            fc_layers.append(nn.Dropout(dropout_rate))

        fc_layers.append(nn.Linear(self.fc_output_size, self.fc_output_size))
        fc_layers.append(nn.ReLU())

        if dropout_rate > 0:
            fc_layers.append(nn.Dropout(dropout_rate))

        self.fc = nn.Sequential(*fc_layers)

        # Distribution Layers
        distribution_layers = [
            nn.Linear(self.fc_output_size, 2*latent_dim) #For mu & sigma
        ]

        self.distribution_layer = nn.Sequential(*distribution_layers)
        
        # Decoder architecture using ConvTranspose2d layers to upsample and reconstruct the image
        decoder_layers = [
            nn.ConvTranspose2d(self.feature_map_size, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        ]

        if dropout_rate > 0:
            decoder_layers.append(nn.Dropout(dropout_rate))

        decoder_layers.extend([
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        ])

        if dropout_rate > 0:
            decoder_layers.append(nn.Dropout(dropout_rate))

        decoder_layers.extend([
            nn.Conv2d(128, img_channels, kernel_size=3, padding=1),  # Output layer (28x28, 1 channel)
            nn.Sigmoid()  # Ensure output is between [0, 1]
        ])

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, z):
        """
        Forward pass of the auto-decoder.

        :param z: the latent vector for the sample
        :return: the reconstructed image
        """
        # Expand latent vector to feature map
        z = self.fc(z)

        z = self.distribution_layer(z) # mu & sigma

        # Generate Normal noise N(0,1)
        noise = torch.normal(0, 1, size=(z.size[0], self.latent_dim), requires_grad=True, device=device)

        # Corresponding Latent Vector:
        z = noise * z[..., self.latent_dim : ] + z[..., : self.latent_dim]
        
        # Reshape to (batch_size, feature_map_size, initial_map_size, initial_map_size)
        z = z.view(-1, self.feature_map_size, self.initial_map_size, self.initial_map_size)
        # Pass through decoder to upsample and reconstruct the image
        z = self.decoder(z)

        return z
