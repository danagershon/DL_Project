import os


def get_next_results_dir(base_dir="results"):
    """
    Get the next available results directory (results1, results2, etc.) inside the base_dir.
    
    :param base_dir: The base directory where the resultsX directories are created.
    :return: The path to the next results directory.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "results1")
    
    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("results")]
    if existing_dirs:
        existing_indices = [int(d.replace("results", "")) for d in existing_dirs if d.replace("results", "").isdigit()]
        next_index = max(existing_indices) + 1
    else:
        next_index = 1
    
    next_dir = os.path.join(base_dir, f"results{next_index}")
    os.makedirs(next_dir)
    return next_dir


def get_latest_results_dir(base_dir="results"):
    """
    Get the latest results directory (resultsX) inside the base_dir.
    :param base_dir: The base directory where the resultsX directories are created.
    :return: The path to the latest results directory or None if no directories exist.
    """
    if not os.path.exists(base_dir):
        print(f"No results directory found in {base_dir}.")
        return None

    existing_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("results")]
    if existing_dirs:
        existing_indices = [int(d.replace("results", "")) for d in existing_dirs if d.replace("results", "").isdigit()]
        latest_index = max(existing_indices)
        return os.path.join(base_dir, f"results{latest_index}")
    else:
        print(f"No valid results directories found in {base_dir}.")
        return None


def save_hyperparameters(hyperparameters, output_dir):
    """
    Save the hyperparameters to a text file in the output directory.
    
    :param hyperparameters: Dictionary containing the hyperparameters.
    :param output_dir: Directory where the hyperparameters should be saved.
    """
    hyperparameters_file = os.path.join(output_dir, "hyperparameters.txt")
    with open(hyperparameters_file, 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")


def save_model_architecture(model, output_dir):
    """
    Save the model architecture (layers) to a text file in the output directory.
    
    :param model: The model whose architecture should be saved.
    :param output_dir: Directory where the architecture should be saved.
    """
    architecture_file = os.path.join(output_dir, "model_architecture.txt")
    with open(architecture_file, 'w') as f:
        f.write(str(model))