import torch
import random
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from tqdm.auto import tqdm
from typing import Tuple

def resample_testmask(test_mask, 
                      p=0.5):
    sample_size = int(np.floor(test_mask.sum()*p))
    true_indices = [i for i, val in enumerate(test_mask) if val]

    sampled_indices = random.sample(true_indices, min(sample_size, len(true_indices)))

    output_tensor = torch.zeros_like(test_mask, dtype=torch.bool)
    output_tensor[sampled_indices] = True

    return output_tensor

def evaluate(data, 
             model, 
             test_mask, 
             percentile_q_list = [99], 
             n_samples=100, 
             device = "cpu",  
             loader = None) -> Tuple[list, list, dict, dict, dict]:
    AUC_list = []
    AP_list = []

    precision_dict = dict()
    recall_dict = dict()
    F1_dict = dict()
    for percentile_q in percentile_q_list:
        precision_dict[percentile_q] = []
        recall_dict[percentile_q] = []
        F1_dict[percentile_q] = []

    model.eval()

    for _ in tqdm(range(n_samples)):
        test_mask_new = resample_testmask(test_mask)
        if loader is None:
            model.eval()
            y_hat = model(data.x, data.edge_index.to(device))
            y_hat = y_hat[test_mask_new].to(device)
            y = data.y[test_mask_new].to(device)
            
        else:
            batch = next(iter(loader))
            batch = batch.to(device, 'edge_index')
            y_hat = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size]

        y_hat  = y_hat.softmax(dim=1) # Get probability of fraud
        
        AUC = roc_auc_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])
        AP = average_precision_score(y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()[:,1])

        AUC_list.append(AUC)
        AP_list.append(AP)

        for percentile_q in percentile_q_list:
            cutoff = np.percentile(y_hat.cpu().detach().numpy()[:,1], percentile_q)
            y_hat_hard = (y_hat[:,1] >= cutoff)*1
            precision = precision_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
            recall = recall_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())
            F1 = f1_score(y.cpu().detach().numpy(), y_hat_hard.cpu().detach().numpy())

            precision_dict[percentile_q].append(precision)
            recall_dict[percentile_q].append(recall)
            F1_dict[percentile_q].append(F1)

    return (AUC_list, AP_list, precision_dict, recall_dict, F1_dict)
