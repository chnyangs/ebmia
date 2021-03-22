"""
    Utility functions for training one epoch
    and evaluating one epoch
"""
import os.path
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from utils.Metrics import accuracy_TU as accuracy

"""
    For GCNs
"""


def train_epoch_sparse(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0
    epoch_train_acc = 0
    nb_data = 0
    gpu_mem = 0
    count = 0
    for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        batch_scores = model.forward(batch_graphs, batch_x, batch_e)
        loss = model.loss(batch_scores, batch_labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        epoch_train_acc += accuracy(batch_scores, batch_labels)
        nb_data += batch_labels.size(0)
        count = iter
    epoch_loss /= (count + 1)
    epoch_train_acc /= nb_data

    return epoch_loss, epoch_train_acc, optimizer


def evaluate_network_sparse(model, device, data_loader, save_info):
    save_flag = save_info['save']
    save_dir = save_info['save_dir']
    save_type = 1 if save_info['data_type'] == 'train' else 0

    model.eval()
    epoch_test_loss = 0
    epoch_test_acc = 0
    nb_data = 0
    train_posterior = []
    train_labels = []
    flag = []

    with torch.no_grad():
        for iter, (batch_graphs, batch_labels) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_labels = batch_labels.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e)

            # Calculate Posteriors
            if save_flag:
                for posterior in F.softmax(batch_scores, dim=1).detach().cpu().numpy().tolist():
                    train_posterior.append(posterior)
                    train_labels.append(int(save_type))

            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            epoch_test_acc += accuracy(batch_scores, batch_labels)
            nb_data += batch_labels.size(0)
        epoch_test_loss /= (iter + 1)
        epoch_test_acc /= nb_data
        # Save Posteriors
        if save_flag:
            x_save_path = os.path.join(save_dir, 'X_train_Label_' + str(save_type) + '.pickle')
            y_save_path = os.path.join(save_dir, 'y_train_Label' + str(save_type) + '.pickle')
            pickle.dump(np.array(train_posterior), open(x_save_path, 'wb'))
            pickle.dump(np.array(train_labels), open(y_save_path, 'wb'))
    return epoch_test_loss, epoch_test_acc
