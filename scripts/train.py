import os
import sys
import optuna
import torch
import logging
import yaml
from yaml import safe_load
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import BCDataset
from model import *
from utils.objectives import *

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

if __name__ == "__main__":
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
    task = m_config["model"]["type"]
    
    if task == "GCN":
        logging.info("Graph Convolutional Network (GCN) selected.")
        data = dataset.to_torch_data()
        def wrapped_objective(trial: optuna.Trial):
            return objective_gcn(
                trial,
                graph     = data,
                masks     = (train_mask, val_mask, test_mask),
                device    = device,
                **m_config["model"]["params"],
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
        with open(f"results/GCN/{task}_training_results.txt", "w") as f:
            f.write(f"Task: {task}\n")
            f.write(f"Parameters: {gcn_params}\n")
            f.write(f"AUC_PRC: {gcn_values}\n") 
    else:
        logging.error(f"Unsupported task: {task}. Supported tasks: GCN.")
        sys.exit(1)
        
    logging.info("Training completed successfully.")