import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from utils.DataUtil import get_mem_data, get_selected_clustering_data
from utils.ModelUtil import evaluate_cluster_distance_attack


if __name__ == '__main__':
    target_number = 500
    num_clusters = 50
    num_nonmembers = 30
    # 1. Load domain 1 dataset
    domain_1_path = "out/MNIST/GCN_MNIST_GPU0_11h15m39s_on_Oct_02_2020/"
    domain_2_path = "out/CIFAR10/GCN_CIFAR10_GPU0_13h39m49s_on_Sep_29_2020/"
    X_train_in_as_non_member, y_train_in_as_non_member, \
    X_train_out_as_non_member, y_train_out_as_non_member = get_mem_data(domain_1_path)

    X_train_in_as_target, y_train_in_as_target, \
    X_train_out_as_target, y_train_out_as_target = get_mem_data(domain_2_path)

    # prepare non-member dataset
    X_non_member = np.concatenate((X_train_in_as_non_member, X_train_out_as_non_member), axis=0)
    # prepare target dataset to evaluate
    X_target = np.concatenate((X_train_in_as_target[0:target_number], X_train_out_as_target[0:target_number]), axis=0)
    # calculate maximum distance between two dataset
    max_original_dist, selected_data = get_selected_clustering_data(X_non_member, X_target,
                                                                    num_clusters, num_nonmembers)
    print("original distance between two dataset:{}".format(max_original_dist))
    params = X_target, target_number, selected_data, max_original_dist
    # apply distance based attack and evaluate the performance
    accuracy, precision, recall, f1 = evaluate_cluster_distance_attack(params)
    print("{},{},{},{},{}".format(accuracy, precision, recall, f1, max_original_dist))
