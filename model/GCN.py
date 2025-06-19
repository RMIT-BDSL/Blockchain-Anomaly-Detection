import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(
        self,
        edge_index: Tensor,
        in_channels: int,
        hidden_dim: int,
        embedding_dim: int,
        output_dim: int = 2,
        num_layers: int = 3,
        dropout: float = 0.0,
        batchnorm: bool = False
    ):
        super().__init__()
        self.edge_index   = edge_index
        self.num_layers     = num_layers
        self.dropout = dropout

        if num_layers == 1:
            self.conv1 = GCNConv(in_channels, embedding_dim)
        else:
            self.conv1 = GCNConv(in_channels, hidden_dim)
            self.hidden_convs = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim)
                for _ in range(num_layers - 2)
            ])
            self.conv2 = GCNConv(hidden_dim, embedding_dim)

        self.out = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: Tensor, edge_index: Tensor = None) -> tuple[Tensor, Tensor]:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        if self.num_layers > 1:
            for conv in self.hidden_convs:
                h = conv(h, edge_index)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.conv2(h, edge_index)

        out = self.out(h)
        return out, h