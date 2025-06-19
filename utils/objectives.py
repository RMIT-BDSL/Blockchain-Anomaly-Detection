import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from model.GCN import GCN

def GNN_features(
    graph: Data,
    model: nn.Module,
    lr: float,
    n_epochs: int,
    train_loader: DataLoader = None,
    val_loader: DataLoader = None,
    test_loader: DataLoader = None,
    train_mask: torch.Tensor = None,
    val_mask: torch.Tensor = None,
    test_mask: torch.Tensor = None,
    **kwargs
):
    device = kwargs.get('device', 'cpu')
    if isinstance(device, str):
        device = torch.device(device)
    model = model.to(device)
    graph = graph.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=kwargs.get('weight_decay', 0.0))
    criterion = nn.CrossEntropyLoss()  # Define loss function.
    
    train_losses = []
    val_scores = []
    
    def train_epoch():
        model.train()
        total_loss = 0.0

        if train_loader is None:
            optimizer.zero_grad()
            out, _ = model(graph.x, graph.edge_index.to(device))
            loss = criterion(out[train_mask], graph.y[train_mask])
            loss.backward()
            optimizer.step()
            return loss.item()
        else:
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out, _ = model(batch.x, batch.edge_index.to(device))
                y_hat = out[: batch.batch_size]
                y_true = batch.y[: batch.batch_size]
                loss = criterion(y_hat, y_true)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(train_loader)

    def evaluate(loader, mask):
        model.eval()
        if loader is None:
            with torch.no_grad():
                out, _ = model(graph.x, graph.edge_index)
                logits = out[mask]
                labels = graph.y[mask]
                probs = logits.softmax(dim=1)
        else:
            all_probs = []
            all_labels = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    out, _ = model(batch.x, batch.edge_index)
                    y_hat = out[: batch.batch_size]
                    all_probs.append(y_hat.softmax(dim=1).cpu())
                    all_labels.append(batch.y[: batch.batch_size].cpu())
            probs = torch.cat(all_probs, dim=0)
            labels = torch.cat(all_labels, dim=0)

        try:
            ap = average_precision_score(labels.numpy(), probs.numpy()[:, 1])
        except Exception:
            preds = probs.argmax(dim=1)
            ap = (preds == labels).sum().item() / labels.size(0)
        return ap

    for _ in range(n_epochs):
        loss_train = train_epoch()
        train_losses.append(loss_train)

    ap_test = evaluate(test_loader, test_mask)
    
    plot_path = kwargs.get('plot_path', None)
    if plot_path is not None:
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)

        # Plot train loss and validation AP on two subplots
        epochs = list(range(1, n_epochs + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), tight_layout=True)

        ax1.plot(epochs, train_losses)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Train Loss")
        ax1.set_title("Training Loss vs. Epoch")

        ax2.plot(epochs, val_scores)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation AP/Acc")
        ax2.set_title("Validation Score vs. Epoch")

        fig.savefig(plot_path)
        plt.close(fig)
        
    return ap_test

def objective_gcn(trial, **kwargs):
    """
    Objective function for GCN hyperparameter search with Optuna.
    """
    def _get(name, suggest_fn):
        return kwargs[name] if name in kwargs else suggest_fn()

    graph = kwargs['graph']

    hidden_dim    = _get('hidden_dim',    lambda: trial.suggest_int('hidden_dim',    64,   256))
    embedding_dim = _get('embedding_dim', lambda: trial.suggest_int('embedding_dim', 32,    128))
    num_layers    = _get('num_layers',    lambda: trial.suggest_int('num_layers',     1,     3))
    lr            = _get('lr',            lambda: trial.suggest_float('lr',        1e-2,  1e-1))
    n_epochs      = _get('n_epochs',      lambda: trial.suggest_int('n_epochs',       5,   500))
    dropout       = _get('dropout',       lambda: trial.suggest_float('dropout',     0.0,  0.5))
    in_channels   = graph.num_node_features
    output_dim    = 2
    # batchnorm     = _get('batchnorm',     lambda: trial.suggest_categorical('batchnorm', [True, False]))

    model_gcn = GCN(
        edge_index=graph.edge_index,
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        dropout=dropout
    )
    
    masks = kwargs.get('masks', None)
    ap_loss = GNN_features(graph,
                           model_gcn,
                           lr,
                           n_epochs,
                           train_mask=masks[0],
                           val_mask=masks[1],
                           test_mask=masks[2]
                           )
    return ap_loss