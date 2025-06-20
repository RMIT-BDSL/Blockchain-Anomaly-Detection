import torch
import os
import sys
import logging
import yaml
import argparse
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import BCDataset
from yaml import safe_load
from model import GCN
from utils.evaluate import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Test a model on the BC dataset.")
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
    with open("config/testing.yaml", "r") as f:
        t_config = safe_load(f)
    with open("config/model.yaml", "r") as f:
        m_config = safe_load(f)

    logging.info("Starting testing process...")
    logging.info("Testing configuration:\n%s", yaml.dump(t_config, sort_keys=False))
    logging.info("Model configuration:\n%s", yaml.dump(m_config, sort_keys=False))

    dataset = BCDataset(
        type=t_config["dataset"]["type"],
        **t_config["dataset"]["kwargs"]
    )

    checkpoint = f"{t_config['pretrained']['checkpoint']}/task/checkpoint.pt"
    task = m_config["model"]["type"]
    result_path = f"{t_config['results']['path']}/{task}"
    device = t_config["device"]
    if isinstance(device, str):
        device = torch.device(device)
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")
    train_mask, val_mask, test_mask = dataset.get_masks()
    percentiles = t_config["evaluation"]["percentiles"]
    data = dataset.to_torch_data()

    if task == "GCN":
        model = GCN(
            in_channels=data.num_features,
            out_channels=dataset.num_classes,
            **m_config["model"]["kwargs"]
        ).to(device)
    else:
        raise ValueError(f"Unsupported model type: {task}")
    logging.info(f"Loading model from {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    logging.info("Model loaded successfully.")

    logging.info("Starting evaluation...")
    results = evaluate(
        model=model,
        data=data,
        test_mask=test_mask,
        percentile_q_list=percentiles,
        device=device
    )
    auc_list, ap_list, precision_dict, recall_dict, f1_dict = results
    logging.info("Evaluation completed.")

    logging.info(f"AUC: {auc_list}")
    logging.info(f"AP: {ap_list}")
    for percentile in percentiles:
        logging.info("" + "="*40)
        logging.info(f"Precision at {percentile}th percentile: {precision_dict[percentile]}")
        logging.info(f"Recall at {percentile}th percentile: {recall_dict[percentile]}")
        logging.info(f"F1 Score at {percentile}th percentile: {f1_dict[percentile]}")

    logging.info("Testing process completed successfully.")

    # Save results TI
    os.makedirs(f"{result_path}", exist_ok=True)
    res_TI = {
        'AUC': auc_list,
        'AP': ap_list
    }
    df_TI = pd.DataFrame(res_TI)
    df_TI.to_csv(os.path.join(result_path, "results_TI.csv"), index=False)

    # save results TD
    res_TD = dict()
    for key in precision_dict.keys():
        res_TD['Precision_' + str(key)] = precision_dict[key]
        res_TD['Recall_' + str(key)] = recall_dict[key]
        res_TD['F1_Score_' + str(key)] = f1_dict[key]
    df_TD = pd.DataFrame(res_TD)
    df_TD.to_csv(os.path.join(result_path, "results_TD.csv"), index=False)