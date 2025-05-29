import pandas as pd
import torch
from torch_geometric.data import Data

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
        
        if self.type == "elliptic":
            self._load_elliptic(**kwargs)
        elif self.type == "ibm":
            self._load_ibm(**kwargs)
        else:
            raise ValueError(f"Dataset type '{type}' is not supported.")
    
    def _load_elliptic(self,
                       path: str = "data/elliptic",
                       classes: dict = {'unknown': 2, '1': 1, '2': 0},
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
        # read raw tables
        feat_df  = pd.read_csv(f"{path}/elliptic_txs_features.csv", header=None)
        edge_df  = pd.read_csv(f"{path}/elliptic_txs_edgelist.csv")
        class_df = pd.read_csv(f"{path}/elliptic_txs_classes.csv")
        
        # rename first two columns: txId, time_step
        feat_df = feat_df.rename(columns={0: 'txId', 1: 'time_step'})
        
        # build feature tensor (includes time_step as one feature)
        feat_array = feat_df.loc[:, 'time_step':].values
        self.features = torch.tensor(feat_array, dtype=torch.float)
        
        # map classes and build label tensor
        mapped = class_df['class'].map(classes).astype(int)
        self.labels = torch.tensor(mapped.values, dtype=torch.long)
        
        # build edge_index for PyG: shape [2, num_edges]
        # assume edge_df has columns ['txId1','txId2']
        edges = edge_df[['txId1','txId2']].values.T
        self.edge_index = torch.tensor(edges, dtype=torch.long)
        
        # build masks by time-step, excluding unknown class (mapped == 2)
        time_step = torch.tensor(feat_df['time_step'].values, dtype=torch.long)
        assert len(time_splits) == 2, "time_splits must have exactly two values"
        t0, t1 = time_splits
        
        known = (self.labels != classes.get('unknown', 2))
        self.train_mask =  (time_step <  t0) & known
        self.val_mask   = ((time_step >= t0) & (time_step <  t1)) & known
        self.test_mask  =  (time_step >= t1) & known
    
    def _load_ibm(self, **kwargs):
        raise NotImplementedError("IBM loader not yet implemented.")
    
    def get_pyg_data(self) -> Data:
        """
        Wrap everything into a torch_geometric.data.Data object.
        """
        data = Data(
            x=self.features,
            y=self.labels,
            edge_index=self.edge_index
        )
        data.train_mask = self.train_mask
        data.val_mask   = self.val_mask
        data.test_mask  = self.test_mask
        return data