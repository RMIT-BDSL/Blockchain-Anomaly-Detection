import torch
import torch.nn.functional as F
import yaml
import warnings
import sys
import random

warnings.filterwarnings("ignore")

from tqdm.auto import tqdm
from torch_geometric.utils import to_networkx
import networkx as nx
import logging
from torch_geometric.data import Data

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load configs
model_config = yaml.safe_load(open('config/model.yaml', 'r'))
training_config = yaml.safe_load(open('config/training.yaml', 'r'))

# Device and data
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
data = torch.load(training_config['data']['data_path'], weights_only=training_config['data']['weights_only']).to(device)
logger.info(f"Loaded data: {data.num_nodes} nodes, {data.num_node_features} features per node")

# Subgraph extractor
def extract_connected_subgraph(data, max_nodes=25, max_edges=40):
    G = to_networkx(data, to_undirected=False)
    max_attempts = 100

    for _ in range(max_attempts):
        start_node = random.choice(list(G.nodes))
        visited = set([start_node])
        queue = [start_node]
        edges = set()

        while queue and len(visited) < max_nodes and len(edges) < max_edges:
            current = queue.pop(0)
            for neighbor in G.neighbors(current):
                if neighbor not in visited and len(visited) < max_nodes:
                    visited.add(neighbor)
                    queue.append(neighbor)
                if current in visited and neighbor in visited:
                    edge = tuple(sorted((current, neighbor)))
                    edges.add(edge)
                if len(edges) >= max_edges:
                    break

        if len(visited) >= 2 and len(edges) > 0:
            break

    final_nodes = list(visited)
    node_map = {old: i for i, old in enumerate(final_nodes)}
    filtered_edges = [
        [node_map[u], node_map[v]] for u, v in edges
        if u in node_map and v in node_map
    ]

    edge_index = torch.tensor(filtered_edges, dtype=torch.long).t().contiguous()
    x = data.x[torch.tensor(final_nodes, dtype=torch.long, device=data.x.device)]
    y = data.y[torch.tensor(final_nodes, dtype=torch.long, device=data.x.device)] if hasattr(data, 'y') else None
    return Data(x=x, edge_index=edge_index, y=y)

# Dynamic subgraph sampling
n_samples_required = training_config['sampling']['n_samples']
max_nodes = training_config['sampling']['max_nodes']
max_edges = training_config['sampling']['max_edges']

temp = []
unique_subgraphs = set()

while len(temp) < n_samples_required:
    remaining = n_samples_required - len(temp)
    logger.info(f"Sampling {remaining} more subgraphs with max_nodes={max_nodes}, max_edges={max_edges}")

    for _ in tqdm(range(remaining)):
        sub_data = extract_connected_subgraph(data, max_nodes=max_nodes, max_edges=max_edges)
        edge_index_tuple = tuple(map(tuple, sub_data.edge_index.t().tolist()))

        if edge_index_tuple not in unique_subgraphs:
            unique_subgraphs.add(edge_index_tuple)
            temp.append(sub_data)

    if len(temp) < n_samples_required:
        logger.warning(f"Only {len(temp)} unique subgraphs. Increasing limits...")
        max_nodes *= 2
        max_edges *= 2

# Save results
torch.save(temp, f"data/elliptic/elliptic_subgraphs.pt")
logger.info(f"Successfully saved {len(temp)} unique subgraphs to elliptic_subgraphs.pt")
# -------------------- Save subgraph data --------------------