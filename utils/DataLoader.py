import dgl
import torch
# from data.superpixels import SuperPixDataset
from dgl.data import LegacyTUDataset


def LoadData(DATASET_NAME):
    # if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
    #     return SuperPixDataset(DATASET_NAME)

    # handling for the TU Datasets
    # https://chrsmrrs.github.io/datasets/docs/datasets/
    TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS', 'PROTEINS_full', 'COLLAB', 'MUTAG','NCI1','NCI109','NCI-H23','AIDS','REDDIT-MULTI-5K',
                   'COIL-RAG','COIL-DEL', 'Fingerprint', 'Letter-high', 'github_stargazers', 'IMDB-BINARY', 'IMDB-MULTI']
    if DATASET_NAME in TU_DATASETS:
        return LegacyTUDataset(DATASET_NAME)


class DGLFormDataset(torch.utils.data.Dataset):
    """
        DGLFormDataset wrapping graph list and label list as per pytorch Dataset.
        *lists (list): lists of 'graphs' and 'labels' with same len().
    """

    def __init__(self, *lists):
        assert all(len(lists[0]) == len(li) for li in lists)
        self.lists = lists
        self.graph_lists = lists[0]
        self.graph_labels = lists[1]

    def __getitem__(self, index):
        return tuple(li[index] for li in self.lists)

    def __len__(self):
        return len(self.lists[0])


def format_graph(graphs, labels):
    for graph in graphs:
        graph.ndata['feat'] = graph.ndata['feat'].float()  # dgl 4.0
        if 'feat' not in graph.edata.keys():
            edge_feat_dim = graph.ndata['feat'].shape[1]  # dim same as node feature dim
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), edge_feat_dim)
    return DGLFormDataset(graphs, labels)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels, dtype=torch.long)
