import losses

auto_decoder_hyperparameters = {
    'batch_size': 32,
    'latent_dim': 256,
    'feature_map_size': 256,
    'epochs': 100,
    'lr': 1e-4,
    'reconstruction_loss': losses.reconstruction_loss_BCE,
    'latent_initialization': 'normal',  # Options: 'normal', 'random', 'uniform'
    'latent_reg_loss_lambda': 1e-5,
    'normal_latent_initialization_variance': 0.1,
    'latent_epochs_for_test': 20,
    'latent_lr_for_test': 1e-3,
    'dropout_rate': 0.2  # 0 means no dropout
}

variational_auto_decoder_hyperparameters = {
    'batch_size': 32,
    'input_dim': 64,
    'latent_dim': 256,
    'feature_map_size': 256,
    'epochs': 100,
    'lr': 1e-4,
    'reconstruction_loss': losses.reconstruction_loss_BCE,
    'latent_initialization': 'normal',  # Options: 'normal', 'random', 'uniform'
    'latent_reg_loss_lambda': 1e-5,
    'normal_latent_initialization_variance': 0.1,
    'latent_epochs_for_test': 20,
    'latent_lr_for_test': 1e-3,
    'dropout_rate': 0.2  # 0 means no dropout
}


# for the new VAD implementation
vad_hyperparameters = {
    'batch_size': 32,
    'latent_dim': 256,
    'feature_map_size': 256,
    'epochs': 100,
    'lr': 1e-4,
    'reconstruction_loss': losses.reconstruction_loss_BCE,
    'latent_initialization': 'normal',  # Options: 'normal', 'random', 'uniform'
    'latent_reg_loss_lambda': 1e-5,
    'normal_latent_initialization_variance': 0.1,
    'latent_epochs_for_test': 20,
    'latent_lr_for_test': 1e-3,
    'dropout_rate': 0.2,  # 0 means no dropout
    'kl_weight': 0.001  # exclusive for VAD
}
