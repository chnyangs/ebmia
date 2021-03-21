"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gated_gcn_net import GatedGCNNet
from nets.gcn_net import GCNNet
from nets.gat_net import GATNet
from nets.gin_net import GINNet
from nets.mlp_net import MLPNet


def GatedGCN(net_params):
    return GatedGCNNet(net_params)


def GCN(net_params):
    return GCNNet(net_params)


def GAT(net_params):
    return GATNet(net_params)


def GIN(net_params):
    return GINNet(net_params)


def MLP(net_params):
    return MLPNet(net_params)


def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GCN': GCN,
        'GAT': GAT,
        'GIN': GIN,
        'MLP': MLP,
    }

    return models[MODEL_NAME](net_params)
