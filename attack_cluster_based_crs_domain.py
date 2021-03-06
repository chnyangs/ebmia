import os
import argparse, json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from utils.DataUtil import get_mem_data, get_selected_clustering_data, select_top_k
from utils.ModelUtil import evaluate_cluster_distance_attack
import torch.nn.functional as F
import torch
tf.config.list_physical_devices('GPU')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', required=True, help="source dataset")
    parser.add_argument('--t', required=True, help="targeted dataset")

    args = parser.parse_args()
    domain_source_path = args.s
    domain_target_path = args.t
    # OGBG_PPA_100_57
    # GCN_CIFAR10_GPU0_20h26m05s_on_Sep_28_2020
    # GCN_PROTEINS_full_GPU0_00h28m28s_on_Jan_03_2021
    X_train_in_as_non_member, y_train_in_as_non_member, \
    X_train_out_as_non_member, y_train_out_as_non_member = get_mem_data(domain_source_path)

    X_train_in_as_target, y_train_in_as_target, \
    X_train_out_as_target, y_train_out_as_target = get_mem_data(domain_target_path)
    # prepare non-member dataset
    X_non_member = np.concatenate((X_train_in_as_non_member, X_train_out_as_non_member), axis=0)
    X_non_member = np.array([F.softmax(torch.FloatTensor(x), dim=0).numpy() for x in X_non_member])

    assert X_train_in_as_target.shape[0] == X_train_out_as_target.shape[0]

    # target_number = X_train_in_as_target.shape[0]
    target_number = X_train_in_as_target.shape[0]
    # prepare target dataset to evaluate
    X_target = np.concatenate((X_train_in_as_target[0:target_number], X_train_out_as_target[0:target_number]), axis=0)

    # calculate maximum distance between two dataset
    num_nonmembers = 10
    # print("X_non_member:{} and X_target:{}".format(X_non_member.shape, X_target.shape))
    num_clusters = 30
    if X_non_member.shape[1] != X_target.shape[1]:
        top_k = min(X_non_member.shape[1], X_target.shape[1])
        X_non_member = select_top_k(X_non_member, top_k)
        X_target = select_top_k(X_target, top_k)
    max_original_dist, selected_data = get_selected_clustering_data(X_non_member, X_target,
                                                                    num_clusters, num_nonmembers)

    print("original distance between two dataset:{}".format(max_original_dist))
    params = X_target, target_number, selected_data, max_original_dist
    # apply distance based attack and evaluate the performance
    accuracy, precision, recall, f1 = evaluate_cluster_distance_attack(params)
    print("{},{},{},{},{}".format(accuracy, precision, recall, f1, max_original_dist))
