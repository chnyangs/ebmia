import argparse
import json
import os
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from train.train_graph_classification import train_epoch_sparse, evaluate_network_sparse
from utils.DataLoader import LoadData, format_graph, collate
from utils.DataUtil import slice_data
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
    per_epoch_time = []
    t0 = time.time()
    save_dir = "Target" + '_' + dataset_name + '_' + model_name + '_' + str(round(t0))
    check_point_dir = os.path.join(out_dir, save_dir)
    save_info = {"save": False, "save_dir": check_point_dir, "data_type": ""}

    # Load dataset
    target_train_set, target_val_set, target_test_set = slice_data(dataset, [0.4, 0.2, 0.4])
    model = gnn_model(model_name, net_params)
    print(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_accs, epoch_val_accs = [], []
    print("target train set size:{}".format(len(target_train_set)))
    print("target train set size:{}".format(len(target_val_set)))
    print("target train set size:{}".format(len(target_test_set)))

    target_train_loader = DataLoader(target_train_set, batch_size=params['batch_size'], shuffle=True,
                                     drop_last=False, collate_fn=collate)
    target_val_loader = DataLoader(target_val_set, batch_size=params['batch_size'], shuffle=False,
                                   drop_last=False, collate_fn=collate)
    target_test_loader = DataLoader(target_test_set, batch_size=params['batch_size'], shuffle=False,
                                    drop_last=False, collate_fn=collate)
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                epoch_train_loss, epoch_train_acc, optimizer = train_epoch_sparse(model,
                                                                                  optimizer,
                                                                                  device,
                                                                                  target_train_loader)

                # evaluate model
                epoch_val_loss, epoch_val_acc = evaluate_network_sparse(model, device, target_val_loader, save_info)
                # test model
                epoch_test_loss, epoch_test_acc = evaluate_network_sparse(model, device, target_test_loader, save_info)
                # record loss & acc
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_accs.append(epoch_train_acc)
                epoch_val_accs.append(epoch_val_acc)
                # formalize print information
                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              test_acc=epoch_test_acc)
                per_epoch_time.append(time.time() - start)

                # Saving checkpoint
                if not os.path.exists(check_point_dir):
                    os.makedirs(check_point_dir)

                # update model parameters
                scheduler.step(epoch_val_loss)

        # update save information & start evaluating & testing model
        save_info['save'] = True
        save_info['data_type'] = "test"
        _, test_acc = evaluate_network_sparse(model, device, target_test_loader, save_info)
        save_info['data_type'] = "train"
        _, train_acc = evaluate_network_sparse(model, device, target_train_loader, save_info)
        print("Test Accuracy: {:.4f}".format(test_acc))
        print("Train Accuracy: {:.4f}".format(train_acc))
        print("Convergence Time (Epochs): {:.4f}".format(epoch))
        print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
        print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

        # save entire model
        torch.save(model, '{}.pkl'.format(check_point_dir + "/target_model", ))
        # save evaluate information
        with open(check_point_dir + '/results.txt', 'w') as f:
            f.write("Train Accuracy:{}\n".format(train_acc))
            f.write("Test Accuracy:{}\n".format(test_acc))
            f.write("Epochs:{}".format(params['epochs']))
            f.close()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Target Model Training --- Exiting from training early because of KeyboardInterrupt')
        print('Removing folder:{}'.format(check_point_dir))
        os.removedirs(check_point_dir)
        print('Removing folder done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="Please give a config.json file with "
                                                        "training/model/data/param details")
    parser.add_argument('--dataset', required=True, help="Please give a value for dataset name")
    parser.add_argument('--model', required=True, help="Please give a value for model name")
    parser.add_argument('--batch', default=64, help="Please give a value for model name")
    parser.add_argument('--epochs', default=100, help="Please give a value for model name")
    parser.add_argument('--half', type=bool, default=False, help="Please give a value for model name")

    args = parser.parse_args()

    # load configuration file
    with open(args.config) as f:
        config = json.load(f)
    # setup gpu
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # read args
    out_dir = config['out_dir']
    dataset_name = args.dataset
    model_name = args.model
    epochs = args.epochs
    # read general params & netparams
    params = config[model_name]['params']
    # check batch_size
    if args.batch != 64:
        params['batch_size'] = int(args.batch)
    if args.epochs != 100:
        params['epochs'] = int(epochs)
    net_params = config[model_name]['net_params']
    config['dataset'] = args.dataset
    dataset = LoadData(dataset_name)
    if args.half:
        print("split dataset into half...")
        dataset, _ = random_split(dataset, [len(dataset) // 2, len(dataset) - len(dataset) // 2])
    graphs, labels = map(list, zip(*dataset))
    dataset = format_graph(graphs, labels)
    net_params['in_dim'] = dataset[0][0].ndata['feat'][0].shape[0]
    num_classes = len(np.unique(labels))
    net_params['n_classes'] = num_classes
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])

    # start train
    train_val()
