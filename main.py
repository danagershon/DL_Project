import argparse

import saving_utilities
from hyperparameters import auto_decoder_hyperparameters, vad_hyperparameters
from TrainerClass import ADTrainer, VADTrainer
from EvaluatorClass import EvaluatorAD, EvaluatorVAD


class TrainEvalModel():

    def __init__(self, train: bool, results_dir, is_VAD) -> None:
        self.train = train
        self.is_VAD = is_VAD
        self.model_output_dir = saving_utilities.create_subdirectories(results_dir, is_VAD)
        self.model_filename = 'variational_auto_decoder.pth' if is_VAD else 'auto_decoder.pth'
        self.latent_filename = 'latent_params.pth' if is_VAD else 'latent_vectors.pth'
        self.trainer_cls = VADTrainer if is_VAD else ADTrainer
        self.evaluator_cls = EvaluatorVAD if is_VAD else EvaluatorAD
        self.hyperparameters = vad_hyperparameters if is_VAD else auto_decoder_hyperparameters
    
    def train_model(self):
        trainer = self.trainer_cls(self.model_output_dir, 
                                   self.model_filename, 
                                   self.latent_filename,
                                   **self.hyperparameters)
        trainer.train()

    def load_and_evaluate_model(self):
        evaluator = self.evaluator_cls(self.hyperparameters, 
                                       self.model_filename, 
                                       self.latent_filename, 
                                       self.model_output_dir)
        evaluator.evaluate()

    def invoke(self):
        if self.train:
            self.train_model()
        self.load_and_evaluate_model()


def main(train_flag, invoke_AD, invoke_VAD, results_dir=None):
    """
    Main function to either train and evaluate the model (AD or VAD), or only evaluate the model based on the flags.
    
    :param train_flag: Boolean, if True the model will be trained, otherwise it will only evaluate from saved files.
    :param invoke_AD: Boolean, if True it will train/evaluate AutoDecoder (AD).
    :param invoke_VAD: Boolean, if True it will train/evaluate VariationalAutoDecoder (VAD).
    :param results_dir: The specific 'results/resultsX' directory to load the model from for evaluation. If None, it will load from the latest directory.
    """
    assert invoke_AD or invoke_VAD, "You must specify either -AD or -VAD to run the AutoDecoder or VariationalAutoDecoder."
    assert train_flag or results_dir, "You must specify if to invoke training and evaluation or evaluation only from given results dir"

    if train_flag:
        # Get the next results directory (results/resultsX)
        results_dir = saving_utilities.get_next_results_dir()
    else:
        # evaluation only. If no results directory is given, use the latest one
        results_dir = results_dir or saving_utilities.get_latest_results_dir()  
    if results_dir is None:
        print("No valid results directory found/given. Cannot evaluate.")
        return
    
    if invoke_AD:
        TrainEvalModel(train_flag, results_dir, is_VAD=False).invoke()
    if invoke_VAD:
        TrainEvalModel(train_flag, results_dir, is_VAD=True).invoke()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate the AutoDecoder (AD) or VariationalAutoDecoder (VAD) model.")
    parser.add_argument("--train", action="store_true", help="Set this flag to train the model.")
    parser.add_argument("-AD", action="store_true", help="Set this flag to train/evaluate the AutoDecoder (AD).")
    parser.add_argument("-VAD", action="store_true", help="Set this flag to train/evaluate the VariationalAutoDecoder (VAD).")
    parser.add_argument("--results_dir", type=str, default=None, help="Specify the results directory (e.g., 'results/resultsX') to load for evaluation.")
    
    args = parser.parse_args()
    main(train_flag=args.train, invoke_AD=args.AD, invoke_VAD=args.VAD, results_dir=args.results_dir)
