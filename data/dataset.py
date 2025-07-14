import os
from typing import Literal, Optional

import pandas as pd
import torch
from torch_geometric.data import Data

from utils.ibm import preprocess_ibm


class BCDataset:
    """
    Universal dataset wrapper for fraud datasets.
    Automatically loads features, edges, labels and masks.
    Currently only 'elliptic' is implemented.
    """
    def __init__(self,
                 type: str,
                 **kwargs):
        """
        Args:
            type (str): Dataset to load. Supported: 'elliptic', 'ibm'.
            **kwargs: Forwarded to dataset-specific loader.
        """
        self.type = type.lower()
        transforms = {
            'elliptic': self._load_elliptic,
            'ibm': self._load_ibm   
        }
        if self.type not in transforms:
            raise ValueError(f"Unsupported dataset type: {self.type}. "
                             f"Supported types: {list(transforms.keys())}")
        transforms[self.type](**kwargs)
        
            
    def _load_elliptic(self,
                       path: str = "datasets/elliptic",
                       classes: dict = {'unknown': 2, '1': 1, '2': 0},
                       directed: bool = False,
                       time_splits: list = [30, 40]):
        """
        Load the Elliptic dataset into:
          - self.features   (FloatTensor[N, F])
          - self.labels     (LongTensor[N])
          - self.edge_index (LongTensor[2, E])
          - self.train_mask (BoolTensor[N])
          - self.val_mask   (BoolTensor[N])
          - self.test_mask  (BoolTensor[N])
        """
        feat_df = pd.read_csv(f"{path}/elliptic_txs_features.csv", header=None)
        edge_df = pd.read_csv(f"{path}/elliptic_txs_edgelist.csv")
        class_df = pd.read_csv(f"{path}/elliptic_txs_classes.csv")
        
        feat_df = feat_df.rename(columns={0: 'txId', 1: 'time_step'})
        feat_array = feat_df.loc[:, 'time_step':].values
        self.features = torch.tensor(feat_array, dtype=torch.float)
        if self.features.size()[1] == 0:
            self.features = torch.ones(self.features.shape[0], 1, dtype=torch.float)

        mapped = class_df['class'].map(classes)
        self.labels = torch.tensor(mapped.values, dtype=torch.int64)
        
        nodes = feat_df['txId']
        map_id = {j: i for i, j in enumerate(nodes)}

        edges_df = edge_df[['txId1', 'txId2']].copy()

        # Handle directionality
        if not directed:
            edges_rev = edges_df.rename(columns={'txId1': 'txId2', 'txId2': 'txId1'})
            edges_df = pd.concat([edges_df, edges_rev], ignore_index=True)

        edges_df['txId1'] = edges_df['txId1'].map(map_id)
        edges_df['txId2'] = edges_df['txId2'].map(map_id)

        # Drop invalid and redundant edges
        edges_df = edges_df.dropna().astype(int)
        edges_df = edges_df[edges_df['txId1'] != edges_df['txId2']]  # remove self-loops
        edges_df = edges_df.drop_duplicates().reset_index(drop=True)

        edge_index = torch.tensor(edges_df.values.T, dtype=torch.long)
        self.edge_index = edge_index
        
        time_step = torch.tensor(feat_df['time_step'].values, dtype=torch.long)
        assert len(time_splits) == 2, "time_splits must have exactly two values"
        t0, t1 = time_splits
        
        known = (self.labels != classes.get('unknown', 2))
        self.train_mask = (time_step < t0) & known
        self.val_mask = ((time_step >= t0) & (time_step < t1)) & known
        self.test_mask = (time_step >= t1) & known
    
    def _load_ibm(self, 
                  path: str = "datasets/ibm",
                  scale: Literal['small', 'medium', 'large'] = 'small',
                  num_obs: int = None,
                  num_pieces: int = 1024,
                  split_ratios: list = [0.8, 0.1, 0.1]):
        """
        Load the IBM dataset into:
            - self.features   (FloatTensor[N, F])
            - self.labels     (LongTensor[N])
            - self.edge_index (LongTensor[2, E])
            - self.train_mask (BoolTensor[N])
            - self.val_mask   (BoolTensor[N])
            - self.test_mask  (BoolTensor[N])
        """
        scale_cap = scale.capitalize()
        feats_file = f"{path}/HI-{scale_cap}_Trans.csv"
        edges_file = f"{path}/edges.csv"

        df = pd.read_csv(feats_file)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y/%m/%d %H:%M')
        df.sort_values('Timestamp', inplace=True)
        df = df[df['Account'] != df['Account.1']]

        total_rows = len(df)
        if num_obs is None or num_obs > total_rows:
            num_obs = total_rows
        df = df.tail(num_obs).reset_index(drop=True)
        df.reset_index(inplace=True)

        if not os.path.exists(edges_file):
            preprocess_ibm(num_obs=len(df),
                           scale=scale,
                           num_pieces=num_pieces,
                           default_path=path)

        edge_df = pd.read_csv(edges_file)
        self.edge_index = torch.tensor(
            edge_df[['txId1','txId2']].values.T,
            dtype=torch.long
        )

        df.columns = [
            'txId','Timestamp',
            'From Bank','Account',
            'To Bank','Account.1',
            'Amount Received','Receiving Currency',
            'Amount Paid','Payment Currency',
            'Payment Format','class'
        ]

        df['Day'] = df['Timestamp'].dt.day
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        df = df.drop(columns=['Timestamp'])

        df = pd.get_dummies(
            df,
            columns=['Receiving Currency','Payment Currency','Payment Format'],
            dtype=float
        )

        self.labels = torch.tensor(df['class'].values, dtype=torch.long)
        drop_cols = {'txId','class','From Bank','Account','To Bank','Account.1'}
        feat_cols = [c for c in df.columns if c not in drop_cols]
        self.features = torch.tensor(df[feat_cols].values, dtype=torch.float)

        N = len(df)
        train_end = int(split_ratios[0] * N)
        val_end = train_end + int(split_ratios[1] * N)

        mask = torch.zeros(N, dtype=torch.bool)
        self.train_mask = mask.clone()
        self.train_mask[:train_end] = True
        self.val_mask = mask.clone()
        self.val_mask[train_end:val_end] = True
        self.test_mask = mask.clone()
        self.test_mask[val_end:] = True
    
    def get_masks(self):
        """
        Returns the train, validation, and test masks.
        """
        if not hasattr(self, 'train_mask') or not hasattr(self, 'val_mask') or not hasattr(self, 'test_mask'):
            raise AttributeError("Masks are not defined. Initialize the dataset first.")
        return (
            self.train_mask,
            self.val_mask,
            self.test_mask
        )
        
    def to_torch_data(self) -> Data:
        """
        Convert stored tensors (features, labels, edge_index, masks) into
        a single torch_geometric.data.Data object.
        """
        x = self.features[:, 1:]
        y = self.labels
        edge_index = self.edge_index

        data = Data(x=x, y=y, edge_index=edge_index)

        data.train_mask = self.train_mask.clone().to(torch.bool)
        data.val_mask   = self.val_mask.clone().to(torch.bool)
        data.test_mask  = self.test_mask.clone().to(torch.bool)

        return data
    
class SubgraphDataset(BCDataset):
    """
    Inherits from BCDataset; transforms the node-level graph
    into pooled subgraph features per original node.
    """
    def __init__(self,
                 type: str,
                 hops: int = 32,
                 max_nodes: Optional[int] = None,
                 **kwargs):
        # Load full graph data
        super().__init__(type=type, **kwargs)
        full = self.to_torch_data()
        self.hops = hops
        self.max_nodes = max_nodes
        self.cache_path = kwargs.get('cache_path', 'subgraphs_cache.pt')

        # Prepare or load cached subgraph data
        if os.path.exists(self.cache_path):
            feats, labs, tr, va, te = torch.load(self.cache_path)
        else:
            feats, labs, tr, va, te = self._extract_subgraphs(full)
            torch.save((feats, labs, tr, va, te), self.cache_path)

        # Assign aggregated subgraph features & labels
        self.features   = feats
        self.labels     = torch.tensor(labs, dtype=torch.long)
        self.train_mask = torch.tensor(tr, dtype=torch.bool)
        self.val_mask   = torch.tensor(va, dtype=torch.bool)
        self.test_mask  = torch.tensor(te, dtype=torch.bool)
        self.edge_index = torch.empty((2,0), dtype=torch.long)

    def _extract_subgraphs(self, data: Data):
        N = data.num_nodes
        x_all = data.x
        y_all = data.y
        ei = data.edge_index
        # Build adjacency list
        adj = [[] for _ in range(N)]
        for u,v in ei.t().tolist():
            adj[u].append(v)
            adj[v].append(u)

        feats, labs, tr_mask, va_mask, te_mask = [], [], [], [], []
        for nid in range(N):
            if not (data.train_mask[nid] or data.val_mask[nid] or data.test_mask[nid]):
                continue
            # BFS up to hops
            visited = {nid}
            frontier = {nid}
            for _ in range(self.hops):
                nxt = set()
                for u in frontier:
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            nxt.add(v)
                frontier = nxt
                if not frontier: 
                    break
            nodes = list(visited)
            if self.max_nodes and len(nodes)>self.max_nodes:
                nodes = nodes[:self.max_nodes]

            # Pool node features within subgraph
            x_sub = x_all[nodes].mean(dim=0)
            feats.append(x_sub)
            labs.append(int(y_all[nid].item()))
            tr_mask.append(bool(data.train_mask[nid].item()))
            va_mask.append(bool(data.val_mask[nid].item()))
            te_mask.append(bool(data.test_mask[nid].item()))

        print (f"Extracted {len(feats)} subgraphs with max hops={self.hops} and max nodes={self.max_nodes}")
        print (len(feats), len(labs), len(tr_mask), len(va_mask), len(te_mask))
        return torch.stack(feats), labs, tr_mask, va_mask, te_mask