import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch_geometric.nn import SAGEConv  # type: ignore


class SAGE(nn.Module):
    """
    Multi-layer GraphSAGE (a.k.a. GraphSAGE) Network, aligned with the GCN/GAT API:

    Args:
      - in_channels: int — size of each input node feature
      - hidden_channels: int — size of hidden node embeddings
      - out_channels: int — number of classes / embedding dim at output
      - num_layers: int — total number of message-passing layers (>=1)
      - dropout: float — dropout probability after each hidden layer
      - batchnorm: bool (default=True) — whether to apply BatchNorm1d after each hidden layer
      - aggregator: str (default='mean') — aggregation scheme ('mean', 'max', 'pool', 'lstm')
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float,
        batchnorm: bool = True,
        aggregator: str = 'mean'
    ):
        super(SAGE, self).__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        self.convs = nn.ModuleList()
        self.batchnorm = batchnorm
        self.bns = nn.ModuleList() if batchnorm else None

        # First layer
        first_out = out_channels if num_layers == 1 else hidden_channels
        self.convs.append(
            SAGEConv(in_channels, first_out, aggr=aggregator)
        )
        if batchnorm and num_layers > 1:
            self.bns.append(nn.BatchNorm1d(first_out))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(hidden_channels, hidden_channels, aggr=aggregator)
            )
            if batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Final layer (if more than one)
        if num_layers > 1:
            self.convs.append(
                SAGEConv(hidden_channels, out_channels, aggr=aggregator)
            )

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        # Reset each conv and BN
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()
        # Xavier on any Linear submodules (if present)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # All but last layer
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Final layer + log-softmax
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)