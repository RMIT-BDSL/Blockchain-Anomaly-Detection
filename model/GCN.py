from typing import Union

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network with optional batch normalization and Xavier initialization.

    Args:
      in_channels (int): Number of input features per node.
      hidden_channels (int): Number of hidden units per layer.
      out_channels (int): Number of output classes.
      num_layers (int): Total number of GCNConv layers (>=1).
      dropout (float): Dropout probability.
      batchnorm (bool): Whether to apply BatchNorm1d after each hidden conv. Default: True.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        batchnorm: bool = True
    ):
        super(GCN, self).__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.convs = nn.ModuleList()
        self.batchnorm = batchnorm
        self.bns = nn.ModuleList() if batchnorm else None

        if num_layers == 1:
            self.convs.append(
                GCNConv(in_channels, out_channels, cached=True)
            )
        else:
            self.convs.append(
                GCNConv(in_channels, hidden_channels, cached=True)
            )
            if batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

            for _ in range(num_layers - 2):
                self.convs.append(
                    GCNConv(hidden_channels, hidden_channels, cached=True)
                )
                if batchnorm:
                    self.bns.append(nn.BatchNorm1d(hidden_channels))

            self.convs.append(
                GCNConv(hidden_channels, out_channels, cached=True)
            )

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        # Reset GCNConv layers
        for conv in self.convs:
            conv.reset_parameters()
        # Reset BatchNorm layers
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()
        # Xavier init on any Linear (if present)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm and len(self.bns) > i:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
    
class ContrastiveGCN(nn.Module):
    """
    Contrastive wrapper that initializes its own GCN encoder from config.

    Args:
      in_channels (int)
      hidden_channels (int)
      out_channels (int)
      num_layers (int)
      dropout (float)
      proj_hidden (int): Hidden dimension of projection MLP.
      proj_out (int): Output dimension of projection MLP.
      batchnorm (bool): Whether to use batchnorm in encoder.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        proj_hidden: int,
        proj_out: int,
        batchnorm: bool = True
    ):
        super(ContrastiveGCN, self).__init__()
        self.encoder = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            batchnorm=batchnorm
        )
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.convs[-1].out_channels, proj_hidden),
            nn.ReLU(),
            nn.Linear(proj_hidden, proj_out)
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for m in self.proj:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # encode
        for i, conv in enumerate(self.encoder.convs[:-1]):
            x = conv(x, edge_index)
            if self.encoder.batchnorm and len(self.encoder.bns) > i:
                x = self.encoder.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.encoder.dropout, training=self.training)
        x = self.encoder.convs[-1](x, edge_index)
        z = self.proj(x)
        return F.normalize(z, dim=1)