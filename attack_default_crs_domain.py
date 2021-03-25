import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from utils.DataUtil import get_mem_data, select_top_k
from utils.ModelUtil import evaluate_cluster_distance_attack, mmd_loss

tf.config.list_physical_devices('GPU')

if __name__ == '__main__':
    # 1. Load domain 1 dataset
    domain_1_path = "results/Target_DD_GCN_1616387754"
    # GCN_MNIST_GPU0_11h15m39s_on_Oct_02_2020
    domain_2_path = "out/CIFAR10/GCN_CIFAR10_GPU0_20h26m05s_on_Sep_28_2020"
    # OGBG_PPA_100_57
    # GCN_CIFAR10_GPU0_20h26m05s_on_Sep_28_2020
    X_train_in_as_non_member, y_train_in_as_non_member, \
    X_train_out_as_non_member, y_train_out_as_non_member = get_mem_data(domain_1_path)

    X_train_in_as_target, y_train_in_as_target, \
    X_train_out_as_target, y_train_out_as_target = get_mem_data(domain_2_path)

    # prepare non-member dataset
    X_non_member = np.concatenate((X_train_in_as_non_member, X_train_out_as_non_member), axis=0)
    # select 30 non-member data sample
    idx = list(range(0, len(X_non_member)))
    random.shuffle(idx)
    selected_idx = idx[0:30]
    X_non_member = X_non_member[selected_idx]
    # prepare target dataset to evaluate
    target_number = X_train_in_as_target.shape[0]
    target_number = 1000 if target_number > 1000 else target_number
    X_target = np.concatenate((X_train_in_as_target[0:target_number], X_train_out_as_target[0:target_number]), axis=0)
    # calculate maximum distance between two dataset
    if X_non_member.shape[1] != X_target.shape[1]:
        top_k = min(X_non_member.shape[1], X_target.shape[1])
        X_non_member = select_top_k(X_non_member, top_k)
        X_target = select_top_k(X_target, top_k)
    # max_original_dist, selected_data = get_selected_single_instances(X_non_member, X_target, num_nonmembers)
    default_dist = mmd_loss(tf.convert_to_tensor(X_non_member, dtype=float),
                            tf.convert_to_tensor(X_target, dtype=float), 1)
    print("original distance between two dataset:{}".format(default_dist))
    params = X_target, target_number, X_non_member, default_dist
    # # apply distance based attack and evaluate the performance
    accuracy, precision, recall, f1 = evaluate_cluster_distance_attack(params)
    print("{},{},{},{},{}".format(accuracy, precision, recall, f1, default_dist))
