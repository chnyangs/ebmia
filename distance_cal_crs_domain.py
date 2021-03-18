import os
import numpy as np
from utils.DataUtil import get_mem_data
from utils.ModelUtil import mmd_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

if __name__ == '__main__':
    # 1. Load domain 1 dataset
    domain_1_path = "out/MNIST/GIN_MNIST_GPU0_11h43m53s_on_Jan_22_2021/"
    domain_2_path = "out/CIFAR10/GCN_CIFAR10_GPU0_13h39m49s_on_Sep_29_2020/"
    X_train_in_1, y_train_in_1, X_train_out_1, y_train_out_1 = get_mem_data(domain_1_path)
    X_train_in_2, y_train_in_2, X_train_out_2, y_train_out_2 = get_mem_data(domain_2_path)
    X_1 = np.concatenate((X_train_in_1, X_train_out_1), axis=0)
    num_nonmembers = 30
    # np.random.seed(10)
    selected_list = np.random.choice(range(0, X_1.shape[0]), num_nonmembers)
    selected_data = X_1[selected_list]
    moves_distance_in = 0
    non_member_correct = 0
    member_correct = 0
    wrong = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    target_number = 500
    X_target = np.concatenate((X_train_in_2[0:target_number], X_train_out_2[0:target_number]), axis=0)
    original_dist = mmd_loss(tf.convert_to_tensor(selected_data, dtype=float),
                             tf.convert_to_tensor(X_target, dtype=float), 1)
    # print("original distance between two dataset:{}".format(original_dist))
    data_size = X_target.shape[0]
    for moves_index in range(data_size):
        selected_data_from_target = X_target[moves_index]
        # moves target to non member
        moves_target_to_non_member = np.append(selected_data, [selected_data_from_target], axis=0)
        # delete moves from target
        moves_target_to_non_member_tensor = tf.convert_to_tensor(moves_target_to_non_member, dtype=float)
        X_target_temp = np.delete(X_target, list(range(moves_index)), axis=0)
        X_target_temp = tf.convert_to_tensor(X_target_temp, dtype=float)
        moves_target_to_non_member_dist = mmd_loss(moves_target_to_non_member_tensor, X_target_temp, 1)
        if moves_index < target_number:
            # moves members
            if original_dist >= moves_target_to_non_member_dist:
                member_correct += 1
                TP += 1
            else:
                FN += 1

        else:
            # moves non-members
            if original_dist <= moves_target_to_non_member_dist:
                non_member_correct += 1
                TN += 1
            else:
                FP += 1
    #
    accuracy = (non_member_correct + member_correct) / data_size
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print("{},{},{},{},{}".format(accuracy, precision, recall, f1, original_dist))

