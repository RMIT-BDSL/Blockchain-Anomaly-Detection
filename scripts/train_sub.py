import argparse
import glob
import json
import logging
import os
import sys
import warnings

import optuna
import torch
import yaml
from yaml import safe_load

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import BCDataset, SubgraphDataset
from model import GAT, GCN, SAGE
from utils.objectives import objective_gnn

warnings.filterwarnings("ignore")

MODEL_REGISTRY = {
    "GCN": GCN,
    "GAT": GAT,
    "SAGE": SAGE
}

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

    dataset = SubgraphDataset(
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
    data = dataset.to_torch_data().to(device)
    # data.x = data.x[:, 1:]