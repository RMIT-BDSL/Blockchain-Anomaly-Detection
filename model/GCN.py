from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network with optional batch normalization and Xavier initialization.

    Args:
      in_channels (int): Number of input features per node.
      hidden_dim (int): Number of hidden units per layer.
      out_channels (int): Number of output classes.
      num_layers (int): Total number of GCNConv layers (>=1).
      dropout (float): Dropout probability.
      batchnorm (bool): Whether to apply BatchNorm1d after each hidden conv. Default: True.
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
        # batchnorm: bool = True
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
            batchnorm (bool): Whether to apply BatchNorm1d after each hidden conv. Default: True.
        """
        super(GCN, self).__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.convs = nn.ModuleList()
        # self.bns = nn.ModuleList() if batchnorm else None
        self.edge_index = edge_index

        if num_layers == 1:
            self.convs.append(
                GCNConv(in_channels, embedding_dim, cached=True)
            )
        else:
            self.convs.append(
                GCNConv(in_channels, hidden_dim, cached=True)
            )
            # if batchnorm:
            #     self.bns.append(nn.BatchNorm1d(hidden_dim))

            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_dim, hidden_dim, cached=True)
                )
                # if batchnorm:
                #     self.bns.append(nn.BatchNorm1d(hidden_dim))

            self.convs.append(
                GCNConv(hidden_dim, embedding_dim, cached=True)
            )

        self.dropout = dropout
        self.out = nn.Linear(embedding_dim, output_dim) if output_dim > 0 else None
        # self.batchnorm = batchnorm
        self.reset_parameters()

    def reset_parameters(self):
        # Reset GCNConv layers
        for conv in self.convs:
            conv.reset_parameters()
        # Reset BatchNorm layers
        # if self.batchnorm:
        #     for bn in self.bns:
        #         bn.reset_parameters()
        # Xavier initialization
        nn.init.xavier_uniform_(self.out.weight) if self.out else None
        if self.out and self.out.bias is not None:
            nn.init.zeros_(self.out.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:
                # if self.batchnorm:
                #     x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        h = x
        out = self.out(h)
        # out = F.log_softmax(out, dim=1) doing cross entropy loss outside

        return out, h