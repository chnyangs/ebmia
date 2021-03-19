import os
import pickle
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf

from utils.ModelUtil import mmd_loss


def load_pickled_data(path):
    with open(path, 'rb') as f:
        unPickler = pickle.load(f)
        return unPickler


def get_all_probs_path(root_path):
    datasets = os.listdir(root_path)
    dataset_models = {}
    if len(datasets) > 0:
        for dataset in datasets:
            models = os.listdir(os.path.join(root_path, dataset))
            dataset_models[dataset] = models
    return dataset_models


def get_mem_data(exp_path):
    X_train_in = load_pickled_data(exp_path + '/X_train_Label_1.pickle')
    y_train_in = load_pickled_data(exp_path + '/y_train_Label_1.pickle')
    X_train_out = load_pickled_data(exp_path + '/X_train_Label_0.pickle')
    y_train_out = load_pickled_data(exp_path + '/y_train_Label_0.pickle')
    return X_train_in, y_train_in, X_train_out, y_train_out


def kmeans_generation(X, num_cluster):
    kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X)
    label_idx = {}
    for label in range(num_cluster):
        index = np.where(kmeans.labels_ == label)[0]
        label_idx[label] = index.tolist()
    return label_idx


def get_selected_clustering_data(X_nm, X_target, num_clusters=10, num_nonmembers=20):
    max_original_dist = 0
    selected_data = np.array([])
    labels = kmeans_generation(X_nm, num_clusters)
    for i in range(num_clusters):
        X_tmp = X_nm[labels[i]]
        # np.random.seed(10)
        selected_list = np.random.choice(range(0, X_tmp.shape[0]), num_nonmembers)
        tmp_data = X_tmp[selected_list]

        tmp_dist = mmd_loss(tf.convert_to_tensor(tmp_data, dtype=float),
                            tf.convert_to_tensor(X_target, dtype=float), 1)
        if tmp_dist > max_original_dist:
            max_original_dist = tmp_dist
            selected_data = tmp_data
    return max_original_dist, selected_data


if __name__ == '__main__':
    data_path = "../out/CIFAR10/GCN_CIFAR10_GPU0_13h39m49s_on_Sep_29_2020/"
    X_train_in, y_train_in, X_train_out, y_train_out = get_mem_data(data_path)
    label_idx = kmeans_generation(X_train_in, 10)
    print(label_idx)
