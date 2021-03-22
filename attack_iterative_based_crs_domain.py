import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import numpy as np
from utils.DataUtil import get_mem_data, get_selected_single_instances
from utils.ModelUtil import evaluate_cluster_distance_attack
tf.config.list_physical_devices('GPU')


if __name__ == '__main__':
    target_number = 500
    # 1. Load domain 1 dataset
    domain_1_path = "results/Target_DD_GCN_1616387754"
    # GCN_MNIST_GPU0_11h15m39s_on_Oct_02_2020
    domain_2_path = "results/Target_PROTEINS_full_GCN_1616388310"
    # GCN_CIFAR10_GPU0_13h39m49s_on_Sep_29_2020
    X_train_in_as_non_member, y_train_in_as_non_member, \
    X_train_out_as_non_member, y_train_out_as_non_member = get_mem_data(domain_1_path)

    X_train_in_as_target, y_train_in_as_target, \
    X_train_out_as_target, y_train_out_as_target = get_mem_data(domain_2_path)

    # prepare non-member dataset
    X_non_member = np.concatenate((X_train_in_as_non_member, X_train_out_as_non_member), axis=0)
    if X_non_member.shape[0] > 2000:
        X_non_member = X_non_member[0:2000]
    # prepare target dataset to evaluate
    X_target = np.concatenate((X_train_in_as_target[0:target_number], X_train_out_as_target[0:target_number]), axis=0)
    # calculate maximum distance between two dataset
    num_nonmembers = 10
    max_original_dist, selected_data = get_selected_single_instances(X_non_member, X_target, num_nonmembers)
    print("original distance between two dataset:{}".format(max_original_dist))
    params = X_target, target_number, selected_data, max_original_dist
    # # apply distance based attack and evaluate the performance
    accuracy, precision, recall, f1 = evaluate_cluster_distance_attack(params)
    print("{},{},{},{},{}".format(accuracy, precision, recall, f1, max_original_dist))
