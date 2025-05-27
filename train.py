import os
import sys
import yaml
import gc
import pandas as pd
import torch
import torch.nn.functional as F
import warnings
import math
import logging
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
from importlib import import_module
from tqdm.auto import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, subgraph
from yaml import safe_load
from sklearn.model_selection import StratifiedKFold
import numpy as np
import model

# one-liner that pulls each symbol from the package into a dict
model_dict = { 
    name: getattr(model, name) for name in model.__all__
}

# -------------------- Setup Logger --------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -------------------- Load Configs --------------------
model_config     = safe_load(open('config/model.yaml',    'r'))
training_config  = safe_load(open('config/training.yaml', 'r'))
optimizer_config = training_config['optimizer']

# -------------------- Load Dataset --------------------
logger.info("Loading Elliptic Dataset...")
with torch.serialization.safe_globals([Data]):
    data = torch.load(
        training_config['data']['data_path'],
        weights_only=training_config['data']['weights_only']
    )
logger.info(f"Nodes: {data.num_nodes}, Features: {data.num_node_features}")

# -------------------- Data Preprocessing --------------------
torch.manual_seed(training_config['training']['seed'])
logger.info(f"Random seed set to: {training_config['training']['seed']}")

mask_known    = (data.y != 2)
known_indices = torch.where(mask_known)[0]
perm          = torch.randperm(len(known_indices))
r0, r1, r2    = training_config['training']['ratio']
n0            = int(len(perm) * r0)
n1            = int(len(perm) * r1)

train_indices = known_indices[perm[:n0]]
valid_indices = known_indices[perm[n0:n0 + n1]]
test_indices  = known_indices[perm[n0 + n1:]]

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
data = data.to(device)

# -------------------- Helper to set masks --------------------
def set_masks(train_idx, val_idx, test_idx=None):
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True
    if test_idx is not None:
        data.test_mask[test_idx] = True

# hold out test permanently
set_masks(train_indices, valid_indices, test_indices)

# -------------------- Cross‐Validation --------------------
logger.info("Starting 5‐fold cross‐validation")
train_val_indices = torch.cat([train_indices, valid_indices])
labels = data.y[train_val_indices].cpu().numpy()

kf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=training_config['training']['seed']
)

cv_accs = []
for fold, (tr, va) in enumerate(kf.split(train_val_indices, labels), start=1):
    fold_train_idx = train_val_indices[tr]
    fold_valid_idx = train_val_indices[va]
    set_masks(fold_train_idx, fold_valid_idx)

    model_cv = model_dict[model_config['model']['type']](
        in_channels=data.num_node_features,
        **model_config['model']['params']
    ).to(device)

    # optimizer & scheduler
    if optimizer_config['type'].lower() == 'adam':
        optimizer_cv = torch.optim.Adam(
            model_cv.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        optimizer_cv = torch.optim.SGD(
            model_cv.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
            momentum=optimizer_config.get('momentum', 0.0)
        )
    scheduler_cv = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_cv, mode='max', factor=0.5, patience=10, verbose=False
    )
    criterion_cv = torch.nn.NLLLoss()

    best_val_acc_cv = 0.0
    early_stop_ctr  = 0
    train_losses_cv, val_losses_cv, val_accs_cv = [], [], []

    ckpt_dir = f"{model_config['checkpoint']}/{model_cv.__class__.__name__}/fold_{fold}"
    os.makedirs(ckpt_dir, exist_ok=True)
    res_dir  = f"results/{model_cv.__class__.__name__}/fold_{fold}"
    os.makedirs(res_dir, exist_ok=True)

    pbar = tqdm(
        range(1, training_config['training']['epochs'] + 1),
        desc=f"Fold {fold} Training",
        dynamic_ncols=True,
        leave=False
    )
    for epoch in pbar:
        model_cv.train()
        optimizer_cv.zero_grad()
        out = model_cv(data.x, data.edge_index)
        loss = criterion_cv(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer_cv.step()
        train_losses_cv.append(loss.item())

        model_cv.eval()
        with torch.no_grad():
            out  = model_cv(data.x, data.edge_index)
            vloss = criterion_cv(out[data.val_mask], data.y[data.val_mask]).item()
            pred  = out[data.val_mask].argmax(dim=1)
            vacc  = (pred == data.y[data.val_mask]).float().mean().item()
        val_losses_cv.append(vloss)
        val_accs_cv.append(vacc)

        scheduler_cv.step(vacc)

        if vacc > best_val_acc_cv:
            best_val_acc_cv = vacc
            early_stop_ctr = 0
            torch.save(
                {'model_state_dict': model_cv.state_dict()},
                f"{ckpt_dir}/{model_cv.__class__.__name__}_fold{fold}.pt"
            )
        else:
            early_stop_ctr += 1
            if early_stop_ctr >= training_config['training']['patience']:
                logger.info(f"Early stopping. Fold {fold} at epoch {epoch:03d}")
                break

        pbar.set_postfix_str(
            f"train_loss={loss:.4f}, val_loss={vloss:.4f}, val_acc={vacc:.4f}"
        )

        gc.collect()
        torch.cuda.empty_cache()

    pbar.close()

    # save plots
    epochs = list(range(1, len(train_losses_cv) + 1))
    plt.figure(figsize=(10,5))
    plt.plot(epochs, train_losses_cv, label='Train Loss')
    plt.plot(epochs, val_losses_cv,   label='Val Loss')
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"Fold {fold} Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{res_dir}/loss_curve.png"); plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(epochs, val_accs_cv, label='Val Acc')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"Fold {fold} Accuracy")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f"{res_dir}/accuracy_curve.png"); plt.close()

    logger.info(f"Fold {fold} best accuracy on validation set: {best_val_acc_cv:.4f}")
    cv_accs.append(best_val_acc_cv)

logger.info("Cross-validation results: " + ", ".join(f"{a:.4f}" for a in cv_accs))
logger.info(f"Mean CV accuracy on validation set: {np.mean(cv_accs):.4f}\n")

# -------------------- Restore test mask --------------------
set_masks([], [], test_indices)

# -------------------- Test each fold --------------------
logger.info("Testing each fold on the held-out test set...")
model = model_dict[model_config['model']['type']](
    in_channels=data.num_node_features,
    **model_config['model']['params']
).to(device)

fold_test_accs = []
for fold in range(1, 6):
    ckpt_path = (
        f"{model_config['checkpoint']}/"
        f"{model.__class__.__name__}/"
        f"fold_{fold}/"
        f"{model.__class__.__name__}_fold{fold}.pt"
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)
        preds = out[data.test_mask].argmax(dim=1)
        acc   = (preds == data.y[data.test_mask]).float().mean().item()

    logger.info(f"Fold {fold} Test Accuracy: {acc:.4f}")
    fold_test_accs.append(acc)

# -------------------- Save test accuracies & configs --------------------
final_txt = f"results/{model.__class__.__name__}/final/test_accuracy.txt"
os.makedirs(os.path.dirname(final_txt), exist_ok=True)
with open(final_txt, 'w') as f:
    for fold, acc in enumerate(fold_test_accs, 1):
        f.write(f"Fold {fold} Test Accuracy: {acc:.4f}\n")

# ----------------- Save model and training configs --------------------
ckpt_base = f"{model_config['checkpoint']}/{model.__class__.__name__}"
os.makedirs(ckpt_base, exist_ok=True)
with open(f"{ckpt_base}/model.yaml",    'w') as f:
    yaml.safe_dump(model_config, f)
with open(f"{ckpt_base}/training.yaml", 'w') as f:
    yaml.safe_dump(training_config, f)

logger.info("Saved test accuracies and configuration files.")
logger.info("Finished cross-validated training and testing.")