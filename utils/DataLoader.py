from data.CSL import CSLDataset
from data.TUs import TUsDataset
from data.superpixels import SuperPixDataset


def LoadData(DATASET_NAME):
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)

        # handling for the TU Datasets
    TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full']
    if DATASET_NAME in TU_DATASETS:
        return TUsDataset(DATASET_NAME)

    if DATASET_NAME == 'CSL':
        return CSLDataset(DATASET_NAME)
