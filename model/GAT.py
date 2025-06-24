import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch_geometric.nn import GATConv, GATv2Conv  # type: ignore


class GAT(nn.Module):
    """
    Multi-layer Graph Attention Network aligned with the GCN API, with Xavier uniform initialization.

    Init args must match GCN:
      - in_channels: int
      - hidden_channels: int
      - out_channels: int
      - num_layers: int
      - dropout: float
      - batchnorm: bool (default True)
    Optional args (with defaults):
      - n_heads: int = 8
      - negative_slope: float = 0.2
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        batchnorm: bool = True,
        n_heads: int = 8,
        negative_slope: float = 0.2
    ):
        super(GAT, self).__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.convs = nn.ModuleList()
        self.batchnorm = batchnorm
        self.bns = nn.ModuleList() if batchnorm else None

        # First layer
        heads = 1 if num_layers == 1 else n_heads
        concat = False if num_layers == 1 else True
        in_c = in_channels
        out_c = out_channels if num_layers == 1 else hidden_channels
        self.convs.append(
            GATConv(
                in_channels=in_c,
                out_channels=out_c,
                heads=heads,
                concat=concat,
                dropout=dropout,
                negative_slope=negative_slope
            )
        )
        if batchnorm and concat:
            feat_dim = out_c * heads
            self.bns.append(nn.BatchNorm1d(feat_dim))

        # Intermediate layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    in_channels=hidden_channels * n_heads,
                    out_channels=hidden_channels,
                    heads=n_heads,
                    concat=True,
                    dropout=dropout,
                    negative_slope=negative_slope
                )
            )
            if batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels * n_heads))

        # Final layer (if more than one)
        if num_layers > 1:
            self.convs.append(
                GATConv(
                    in_channels=hidden_channels * n_heads,
                    out_channels=out_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    negative_slope=negative_slope
                )
            )

        self.dropout = dropout
        # Apply Xavier initialization
        self.reset_parameters()

    def reset_parameters(self):
        # Reset conv and batchnorm parameters
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()
        # Xavier uniform init on all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Apply all but the last layer
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Last layer + log-softmax
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)
    
class GATv2(nn.Module):
    """
    Multi-layer Graph Attention v2 Network, mirroring the GAT API, with Xavier uniform initialization.

    Init args must match GAT:
      - in_channels: int
      - hidden_channels: int
      - out_channels: int
      - num_layers: int
      - dropout: float
      - batchnorm: bool (default True)
    Optional args (with defaults):
      - n_heads: int = 8
      - negative_slope: float = 0.2
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        batchnorm: bool = True,
        n_heads: int = 8,
        negative_slope: float = 0.2
    ):
        super(GATv2, self).__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.convs = nn.ModuleList()
        self.batchnorm = batchnorm
        self.bns = nn.ModuleList() if batchnorm else None

        # First layer
        heads = 1 if num_layers == 1 else n_heads
        concat = False if num_layers == 1 else True
        in_c = in_channels
        out_c = out_channels if num_layers == 1 else hidden_channels
        self.convs.append(
            GATv2Conv(
                in_channels=in_c,
                out_channels=out_c,
                heads=heads,
                concat=concat,
                dropout=dropout,
                negative_slope=negative_slope,
            )
        )
        if batchnorm and concat:
            feat_dim = out_c * heads
            self.bns.append(nn.BatchNorm1d(feat_dim))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_channels * n_heads,
                    out_channels=hidden_channels,
                    heads=n_heads,
                    concat=True,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            )
            if batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels * n_heads))

        # Final layer
        if num_layers > 1:
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_channels * n_heads,
                    out_channels=out_channels,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    negative_slope=negative_slope,
                )
            )

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        # Reset conv and batchnorm params
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()
        # Xavier init on all linears
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)