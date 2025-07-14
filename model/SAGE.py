from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GraphNorm, SAGEConv


class SAGE(nn.Module):
    """
    Multi-layer GraphSAGE Network with optional GraphNorm and Xavier initialization.

    Args:
      edge_index (Tensor): Edge indices of the graph.
      in_channels (int): Number of input features per node.
      hidden_dim (int): Number of hidden units per layer.
      embedding_dim (int): Dimension of the output embeddings.
      output_dim (int): Number of output classes (0 for no output layer).
      num_layers (int): Total number of SAGEConv layers (>=1).
      dropout (float): Dropout probability.
      graphnorm (bool): Whether to apply GraphNorm after each hidden conv. Default: True.
      aggregator (str): Aggregation function ('mean', 'max', 'pool', 'lstm'). Default: 'mean'.
    """
    def __init__(
        self,
        edge_index: Tensor,
        in_channels: int,
        hidden_dim: int,
        embedding_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        graphnorm: bool = False,
        aggregator: str = "mean"
    ) -> None:
        super(SAGE, self).__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList() if graphnorm else None
        self.edge_index = edge_index
        self.dropout = dropout
        self.graphnorm = graphnorm

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, embedding_dim, aggr=aggregator))
        else:
            # Input layer
            self.convs.append(SAGEConv(in_channels, hidden_dim, aggr=aggregator))
            if graphnorm:
                self.gns.append(GraphNorm(hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
                if graphnorm:
                    self.gns.append(GraphNorm(hidden_dim))

            # Output layer
            self.convs.append(SAGEConv(hidden_dim, embedding_dim, aggr=aggregator))

        self.out = nn.Linear(embedding_dim, output_dim) if output_dim > 0 else None
        # self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.graphnorm:
            for gn in self.gns:
                gn.reset_parameters()
        if self.out:
            nn.init.xavier_uniform_(self.out.weight)
            if self.out.bias is not None:
                nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                if self.graphnorm:
                    x = self.gns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.out(x) if self.out else x
        return out