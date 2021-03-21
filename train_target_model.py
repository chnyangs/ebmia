import argparse
import json
import os
import numpy as np
import torch
from torch import optim

from utils.DataLoader import LoadData, format_graph
from utils.NetUtil import gnn_model


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def train_val():
    # Load dataset
    print(len(dataset))
    t_model = gnn_model(model_name, net_params)
    t_model = t_model.to(device)

    t_optimizer = optim.Adam(t_model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    t_scheduler = optim.lr_scheduler.ReduceLROnPlateau(t_optimizer, mode='min',
                                                       factor=params['lr_reduce_factor'],
                                                       patience=params['lr_schedule_patience'],
                                                       verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Please give a config.json file with "
                                                        "training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--model', help="Please give a value for model name")

    args = parser.parse_args()

    # load configuration file
    with open(args.config) as f:
        config = json.load(f)
    # setup gpu
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # read general params
    params = config['params']
    model_name = config['model']
    dataset_name = config['dataset']
    dataset = LoadData(dataset_name)
    graphs, labels = map(list, zip(*dataset))
    formated_dataset = format_graph(graphs, labels)
    # read net params
    net_params = config['net_params']
    net_params['in_dim'] = formated_dataset[0][0].ndata['feat'][0].shape[0]
    num_classes = len(np.unique(labels))
    net_params['n_classes'] = num_classes
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    dirs = 'save'
    train_val()
