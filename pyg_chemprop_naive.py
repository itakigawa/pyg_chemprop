import torch
import torch.nn as nn
import torch.utils.data
from torch_geometric.nn import global_add_pool


def directed_mp(message, edge_index):
    m = []
    for k, (i, j) in enumerate(zip(*edge_index)):
        edge_to_i = (edge_index[1] == i)
        edge_from_j = (edge_index[0] == j)
        nei_message = message[edge_to_i]
        rev_message = message[edge_to_i & edge_from_j]
        m.append(torch.sum(nei_message, dim=0) - rev_message)
    return torch.vstack(m)


def aggregate_at_nodes(x, message, edge_index):
    out_message = []
    for i in range(x.shape[0]):
        edge_to_i = (edge_index[1] == i)
        out_message.append(torch.sum(message[edge_to_i], dim=0))
    return torch.vstack(out_message)


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden_size, node_fdim, edge_fdim, depth=3):
        super(DMPNNEncoder, self).__init__()
        self.act_func = nn.ReLU()
        self.W1 = nn.Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(node_fdim + hidden_size, hidden_size, bias=False)
        self.depth = depth

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        input = self.W1(init_msg)
        h0 = self.act_func(input)

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(x, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global sum pooling
        return global_add_pool(node_attr, batch)
