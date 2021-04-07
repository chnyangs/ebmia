import os
import argparse, json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from utils.DataUtil import get_mem_data, get_selected_single_instances, select_top_k
from utils.ModelUtil import evaluate_cluster_distance_attack
tf.config.list_physical_devices('GPU')


if __name__ == '__main__':
    # 1. Load domain 1 dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', required=True, help="source dataset")
    parser.add_argument('--t', required=True, help="targeted dataset")

    args = parser.parse_args()
    domain_source_path = args.s
    domain_target_path = args.t
    # GCN_CIFAR10_GPU0_13h39m49s_on_Sep_29_2020
    X_train_in_as_non_member, y_train_in_as_non_member, \
    X_train_out_as_non_member, y_train_out_as_non_member = get_mem_data(domain_source_path)

    X_train_in_as_target, y_train_in_as_target, \
    X_train_out_as_target, y_train_out_as_target = get_mem_data(domain_target_path)

    # prepare non-member dataset
    X_non_member = np.concatenate((X_train_in_as_non_member, X_train_out_as_non_member), axis=0)
    # prepare target dataset to evaluate
    target_number = min(X_train_in_as_target.shape[0], X_train_out_as_target.shape[0])
    target_number = 1000 if target_number > 1000 else target_number
    if X_non_member.shape[0] > target_number * 2:
        X_non_member = X_non_member[0:target_number * 2]
    X_target = np.concatenate((X_train_in_as_target[0:target_number], X_train_out_as_target[0:target_number]), axis=0)
    if X_non_member.shape[1] != X_target.shape[1]:
        print(X_non_member.shape, X_target.shape)
        top_k = min(X_non_member.shape[1], X_target.shape[1])
        X_non_member = select_top_k(X_non_member, top_k)
        X_target = select_top_k(X_target, top_k)
    # calculate maximum distance between two dataset
    num_nonmembers = 10
    max_original_dist, selected_data = get_selected_single_instances(X_non_member, X_target, num_nonmembers)
    print("original distance between two dataset:{}".format(max_original_dist))
    params = X_target, target_number, selected_data, max_original_dist
    # # apply distance based attack and evaluate the performance
    accuracy, precision, recall, f1 = evaluate_cluster_distance_attack(params)
    print("{},{},{},{},{}".format(accuracy, precision, recall, f1, max_original_dist))
