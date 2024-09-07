import argparse
import training
import evaluation_utils
import saving_utilities
from hyperparameters import auto_decoder_hyperparameters


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
        saving_utilities.save_hyperparameters(auto_decoder_hyperparameters, results_dir)
        # Train the model and save pth files in the output_dir
        training.train_auto_encoder(output_dir=results_dir, **auto_decoder_hyperparameters)
    
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