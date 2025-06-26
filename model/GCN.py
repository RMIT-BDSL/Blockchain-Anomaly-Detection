from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, GraphNorm


class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network with optional batch normalization and Xavier initialization.

    Args:
      in_channels (int): Number of input features per node.
      hidden_dim (int): Number of hidden units per layer.
      out_channels (int): Number of output classes.
      num_layers (int): Total number of GCNConv layers (>=1).
      dropout (float): Dropout probability.
      graphnorm (bool): Whether to apply GraphNorm after each hidden conv. Default: True.
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
        graphnorm: bool = False
    ) -> None:
        """
        Initialize the GCN model.
        Args:
            edge_index (Tensor): Edge indices of the graph.
            in_channels (int): Number of input features per node.
            hidden_dim (int): Number of hidden units per layer.
            embedding_dim (int): Dimension of the output embeddings.
            output_dim (int): Number of output classes (0 for no output layer).
            num_layers (int): Total number of GCNConv layers (>=1).
            dropout (float): Dropout probability.
            graphnorm (bool): Whether to apply GraphNorm after each hidden conv. Default: True.
        """
        super(GCN, self).__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.convs = nn.ModuleList()
        self.gns = nn.ModuleList() if graphnorm else None
        self.edge_index = edge_index

        if num_layers == 1:
            self.convs.append(
                GCNConv(in_channels, embedding_dim, cached=True)
            )
        else:
            self.convs.append(
                GCNConv(in_channels, hidden_dim, cached=True)
            )
            if graphnorm:
                self.gns.append(GraphNorm(hidden_dim))

            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_dim, hidden_dim, cached=True)
                )
                if graphnorm:
                    self.gns.append(GraphNorm(hidden_dim))

            self.convs.append(
                GCNConv(hidden_dim, embedding_dim, cached=True)
            )

        self.dropout = dropout
        self.out = nn.Linear(embedding_dim, output_dim) if output_dim > 0 else None
        self.graphnorm = graphnorm
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        if self.graphnorm:
            for gn in self.gns:
                gn.reset_parameters()

        # Xavier initialization
        nn.init.xavier_uniform_(self.out.weight) if self.out else None
        if self.out and self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                if self.graphnorm:
                    x = self.gns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        out = self.out(x) if self.out else x
        # out = F.log_softmax(out, dim=1) doing cross entropy loss outside

        return out