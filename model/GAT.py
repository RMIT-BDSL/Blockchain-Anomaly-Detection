from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch_geometric.nn import GATConv, GraphNorm


class GAT(nn.Module):
    """
    Multi-layer Graph Attention Network with optional GraphNorm and Xavier initialization.

    Args:
      edge_index (Tensor): Edge indices of the graph.
      in_channels (int): Number of input features per node.
      hidden_dim (int): Number of hidden units per layer.
      embedding_dim (int): Dimension of the output embeddings.
      output_dim (int): Number of output classes (0 for no output layer).
      num_layers (int): Total number of GATConv layers (>=1).
      dropout (float): Dropout probability.
      graphnorm (bool): Whether to apply GraphNorm after each hidden conv. Default: True.
      n_heads (int): Number of attention heads. Default: 8.
      negative_slope (float): LeakyReLU negative slope. Default: 0.2.
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
        n_heads: int = 4,
        negative_slope: float = 0.2
    ) -> None:
        super(GAT, self).__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList() if graphnorm else None
        self.edge_index = edge_index
        self.dropout = dropout
        self.graphnorm = graphnorm
        self.n_heads = n_heads

        if num_layers == 1:
            self.convs.append(
                GATConv(in_channels, embedding_dim, heads=1, concat=False, dropout=dropout, negative_slope=negative_slope)
            )
        else:
            self.convs.append(
                GATConv(in_channels, hidden_dim, heads=n_heads, concat=True, dropout=dropout, negative_slope=negative_slope)
            )
            if graphnorm:
                self.gns.append(GraphNorm(hidden_dim * n_heads))

            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_dim * n_heads, hidden_dim, heads=n_heads, concat=True, dropout=dropout, negative_slope=negative_slope)
                )
                if graphnorm:
                    self.gns.append(GraphNorm(hidden_dim * n_heads))

            # Output layer
            self.convs.append(
                GATConv(hidden_dim * n_heads, embedding_dim, heads=1, concat=False, dropout=dropout, negative_slope=negative_slope)
            )

        self.out = nn.Linear(embedding_dim, output_dim) if output_dim > 0 else None
        self.reset_parameters()

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