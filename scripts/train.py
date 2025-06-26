import argparse
import glob
import json
import logging
import os
import sys

import optuna
import torch
import yaml
from yaml import safe_load

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import BCDataset
from utils.objectives import objective_gcn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# optuna.logging.set_verbosity(optuna.logging.DEBUG)

def parse_args():
    # get logging level from command line arguments
    parser = argparse.ArgumentParser(description="Train a model on the BC dataset.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level."
    )
    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level.upper())
    return args

if __name__ == "__main__":
    args = parse_args()
    with open("config/training.yaml", "r") as f:
        t_config = safe_load(f)
    with open("config/model.yaml", "r") as f:
        m_config = safe_load(f)

    logging.info("Starting training process...")
    logging.info("Training configuration:\n%s", yaml.dump(t_config, sort_keys=False))
    logging.info("Model configuration:\n%s", yaml.dump(m_config, sort_keys=False))

    dataset = BCDataset(
        type=t_config["dataset"]["type"],
        **t_config["dataset"]["kwargs"]
    )
    
    device = t_config["device"]
    if isinstance(device, str):
        device = torch.device(device)
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    train_mask, val_mask, test_mask = dataset.get_masks()
    data = dataset.to_torch_data()
    task = m_config["model"]["type"]
    
    if task == "GCN":
        logging.info("Graph Convolutional Network (GCN) selected.")
        def wrapped_objective(trial: optuna.Trial):
            return objective_gcn(
                trial,
                graph     = data,
                masks     = (train_mask, val_mask, test_mask),
                device    = device,
                result_path = 'checkpoints/GCN',
                weight_decay = t_config["optimizer"]["weight_decay"],
                # **m_config["model"]["params"],
            )
        study = optuna.create_study(direction="maximize")
        study.optimize(
            wrapped_objective,
            n_trials=t_config["training"]["n_trials"]
        )
        gcn_params = study.best_params
        gcn_values = study.best_value
        logging.info(f"Best GCN parameters: {gcn_params}, Best value: {gcn_values}")
        
        # save results
        os.makedirs("results/GCN", exist_ok=True)
        logging.info(f"Saving GCN training results for task: {task}")

        with open(f"results/GCN/{task}_training_results.json", "w") as f:
            json.dump({
                "Task": task,
                "Parameters": gcn_params,
                "AUC_PRC": gcn_values,
                "Study": study.trials_dataframe().to_dict(orient="records")
            }, f, indent=4, default=str)

        best_path = study.best_trial.user_attrs["model_state_path"]
        logging.info("Best-trial weights are here: %s", best_path)

        for f in glob.glob("checkpoints/GCN/gcn_trial_*.pt"):
            if f.replace('\\', '/') != best_path and f.endswith(".pt"):
                os.remove(f)

        os.rename(best_path, "checkpoints/GCN/gcn_best.pt")
        logging.info(f"✅ Kept only best checkpoint {best_path}")

    else:
        logging.error(f"Unsupported task: {task}. Supported tasks: GCN.")
        sys.exit(1)

    logging.info("✅ Results saved successfully.")
    logging.info("✅ Training phase completed successfully.")