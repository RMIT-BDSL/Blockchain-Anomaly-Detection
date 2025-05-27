import networkx as nx
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data

edgelists = pd.read_csv('elliptic/elliptic_txs_edgelist.csv')
features = pd.read_csv('elliptic/elliptic_txs_features.csv')
classes = pd.read_csv('elliptic/elliptic_txs_classes.csv')

features.columns = ['txId'] + [f"V{i + 1}" for i in range(len(features.columns) - 1)]
label_map = {'1': 0, '2': 1, 'unknown': 2}
classes['label'] = classes['class'].map(label_map)

unique_txids = features['txId'].unique()
txid_to_idx = {txid: idx for idx, txid in enumerate(unique_txids)}

source_nodes = edgelists['txId1'].map(txid_to_idx)
target_nodes = edgelists['txId2'].map(txid_to_idx)
mask_valid = source_nodes.notna() & target_nodes.notna()
source_nodes_filtered = source_nodes[mask_valid].astype(int)
target_nodes_filtered = target_nodes[mask_valid].astype(int)
edge_index_np = np.array([source_nodes_filtered.values, target_nodes_filtered.values])
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

features_sorted = features.sort_values('txId', key=lambda col: col.map(txid_to_idx))
x = torch.tensor(features_sorted.drop(columns=['txId']).values, dtype=torch.float)

classes = classes[classes['txId'].isin(unique_txids)]
classes_sorted = classes.sort_values('txId', key=lambda col: col.map(txid_to_idx))
y = torch.tensor(classes_sorted['label'].values, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)
mask_known = (data.y != 2)

print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Feature size:", data.num_node_features)
torch.save(data, 'data/elliptic/elliptic_data.pt')
print("Data saved to 'data/elliptic/elliptic_data.pt'")