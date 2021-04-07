import os
import random
import argparse, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from utils.DataUtil import get_mem_data, select_top_k
from utils.ModelUtil import evaluate_cluster_distance_attack, mmd_loss

tf.config.list_physical_devices('GPU')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', required=True, help="source dataset")
    parser.add_argument('--t', required=True, help="targeted dataset")

    args = parser.parse_args()
    domain_source_path = args.s
    domain_target_path = args.t
    # 1. Load domain 1 dataset
    # domain_1_path = "exp1/Target_NCI109_GCN_1617015703"
    # # GCN_MNIST_GPU0_11h15m39s_on_Oct_02_2020
    # domain_2_path = "exp1/Target_AIDS_GCN_1617014492"
    # OGBG_PPA_100_57
    # GCN_CIFAR10_GPU0_20h26m05s_on_Sep_28_2020
    X_train_in_as_non_member, y_train_in_as_non_member, \
    X_train_out_as_non_member, y_train_out_as_non_member = get_mem_data(domain_source_path)

    X_train_in_as_target, y_train_in_as_target, \
    X_train_out_as_target, y_train_out_as_target = get_mem_data(domain_target_path)

    # prepare non-member dataset
    X_non_member = np.concatenate((X_train_in_as_non_member, X_train_out_as_non_member), axis=0)
    # select 30 non-member data sample
    idx = list(range(0, len(X_non_member)))
    random.shuffle(idx)
    selected_idx = idx[0:30]
    X_non_member = X_non_member[selected_idx]
    # prepare target dataset to evaluate
    target_number = min(X_train_in_as_target.shape[0], X_train_out_as_target.shape[0])
    target_number = 2000 if target_number > 2000 else target_number
    assert X_train_in_as_target.shape[0] == X_train_out_as_target.shape[0]
    target_in_idx, target_out_idx = list(range(0, X_train_in_as_target.shape[0])), \
                                    list(range(0, X_train_out_as_target.shape[0]))
    random.shuffle(target_in_idx)
    random.shuffle(target_out_idx)
    selected_target_in_idx = target_in_idx[0:target_number]
    selected_target_out_idx = target_out_idx[0:target_number]
    print("selected_target_in_idx:{} and selected_target_out_idx:{}".format(len(selected_target_in_idx),
                                                                            len(selected_target_out_idx)))
    X_target = np.concatenate((X_train_in_as_target[selected_target_in_idx],
                               X_train_out_as_target[selected_target_out_idx]), axis=0)
    # calculate maximum distance between two dataset
    if X_non_member.shape[1] != X_target.shape[1]:
        top_k = min(X_non_member.shape[1], X_target.shape[1])
        X_non_member = select_top_k(X_non_member, top_k)
        X_target = select_top_k(X_target, top_k)
    print("X_target:{} and X_non_member:{}".format(X_target.shape, X_non_member.shape))
    # max_original_dist, selected_data = get_selected_single_instances(X_non_member, X_target, num_nonmembers)
    default_dist = mmd_loss(tf.convert_to_tensor(X_non_member, dtype=float),
                            tf.convert_to_tensor(X_target, dtype=float), 1)
    print("original distance between two dataset:{}".format(default_dist))
    params = X_target, target_number, X_non_member, default_dist
    # # apply distance based attack and evaluate the performance
    accuracy, precision, recall, f1 = evaluate_cluster_distance_attack(params)
    print("{},{},{},{},{}".format(accuracy, precision, recall, f1, default_dist))
