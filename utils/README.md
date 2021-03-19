
# Dataset

## dgl.data.MiniGCDataset


## dgl.data.GINDataset
    MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, 
    PROTEINS, PTC, REDDITBINARY, REDDITMULTI5K
    

## dgl.data.LegacyTUDataset

    ENZYMES, DD, COLLAB, MUTAG



## dgl.data.TUDataset

    ENZYMES, DD, COLLAB, MUTAG

Examples:

    data = GINDataset(name='MUTAG', self_loop=False)    
    data = LegacyTUDataset('DD') 
    data = TUDataset('DD')
    data = MiniGCDataset(100, 16, 32, seed=0)
    graphs, labels = zip(*[data[i] for i in range(16)])
    batched_graphs = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    batched_graphs

## dgl.data.utils.Subset(dataset, indices)

More Details [View](https://docs.dgl.ai/en/latest/api/python/dgl.data.html#mini-graph-classification-dataset)