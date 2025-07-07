import logging
import pandas as pd
import time
import pickle
from tqdm.auto import tqdm

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parameters
DATAPATH = "./datasets/elliptic2"
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

# Load background nodes
start = time.time()
feat = pd.read_csv(f"{DATAPATH}/background_nodes.csv")
feature_cols = list(feat.columns[1:])
logger.info(f"Loaded node features: {feature_cols}")
logger.info(f"Sample of background nodes:\n{feat.head()}")
logger.info(f"Loaded {len(feat)} background nodes in {time.time() - start:.2f} seconds.")

# Create node ID mapping
start = time.time()
n2id = pd.Series(feat.index.values, index=feat.iloc[:, 0]).to_dict()
maxid = max(n2id.values())
with open('n2id.pkl', 'wb') as fp:
    pickle.dump(n2id, fp)
logger.info(f"Stored node-to-index mapping for {len(n2id)} nodes in {time.time() - start:.2f} seconds.")

# Load background edges
start = time.time()
edges = pd.read_csv(f"{DATAPATH}/background_edges.csv", usecols=["clId1", "clId2"])
logger.info(f"Loaded {len(edges)} edges in {time.time() - start:.2f} seconds.")

# Write edge list with progress bar
start = time.time()
with open("./edge_list.txt", "w") as file:
    for _, row in tqdm(edges.iterrows(), total=len(edges), desc="Writing edge list"):
        try:
            file.write(f"{n2id[row.clId1]} {n2id[row.clId2]}\n")
        except KeyError as e:
            logger.warning(f"Node ID not found in n2id: {e}")
logger.info(f"Stored edge list in {time.time() - start:.2f} seconds.")

# Load subgraph data
start = time.time()
cc = pd.read_csv(f"{DATAPATH}/connected_components.csv")
edge = pd.read_csv(f"{DATAPATH}/edges.csv")  # Not used here, but loaded for completeness
node = pd.read_csv(f"{DATAPATH}/nodes.csv")
logger.info(f"Loaded subgraph metadata in {time.time() - start:.2f} seconds.")

# Map subgraph IDs
cc2id = pd.Series(cc.index.values, index=cc.iloc[:, 0]).to_dict()
logger.info(f"Identified {len(cc2id)} unique subgraphs.")

# Group nodes by subgraph
subgraphs = {}
for node_id, cc_id in tqdm(node.itertuples(index=False), total=len(node), desc="Grouping nodes by subgraph"):
    sid = cc2id[cc_id]
    subgraphs.setdefault(sid, []).append(str(n2id[node_id]))

# Write subgraphs to file with batching and progress bar
start = time.time()
lines = []
for i, (sid, nodes) in enumerate(tqdm(subgraphs.items(), desc="Writing subgraph file")):
    label = cc.loc[sid, "ccLabel"]
    tag = "train" if i % 10 <= 7 else "val" if i % 10 == 8 else "test"
    lines.append(f"{'-'.join(nodes)}\t{label}\t{tag}\n")

with open("./subgraphs.pt", "w") as file:
    file.writelines(lines)
logger.info(f"Wrote subgraph samples to subgraphs.pt in {time.time() - start:.2f} seconds.")