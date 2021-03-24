import os
import pickle
from sklearn.cluster import KMeans
import numpy as np
import tensorflow as tf
from torch.utils.data import random_split
from tqdm import tqdm
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
    for label in tqdm(range(num_cluster),desc="KMeans clustering:"):
        index = np.where(kmeans.labels_ == label)[0]
        label_idx[label] = index.tolist()
    return label_idx


def get_selected_clustering_data(X_nm, X_target, num_clusters=10, num_nonmembers=20):
    max_original_dist = 0
    selected_data = np.array([])
    labels = kmeans_generation(X_nm, num_clusters)
    for i in tqdm(range(num_clusters), desc="calculate cluster distance:"):
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


def get_selected_single_instances(X_nm, X_target, num_nonmembers=10):
    distance_dict = {}
    for i in tqdm(range(X_nm.shape[0]), desc="calculate distance:"):
        tmp_dist = mmd_loss(tf.convert_to_tensor([X_nm[i]], dtype=float),
                            tf.convert_to_tensor(X_target, dtype=float), 1)
        distance_dict[i] = tmp_dist.numpy()
    distance_dict = sorted(distance_dict.items(), key=lambda x: x[1],reverse=True)
    top_index = []
    for dd in distance_dict[0:num_nonmembers]:
        top_index.append(dd[0])
    max_original_dist = mmd_loss(tf.convert_to_tensor(X_nm[top_index], dtype=float),
                                 tf.convert_to_tensor(X_target, dtype=float), 1)
    return max_original_dist, X_nm[top_index]


def slice_data(data, pct=None):
    if pct is None:
        pct = [0.7, 0.3]
    if len(pct) == 2:
        train_size = round(len(data) * pct[0])
        test_size = len(data) - train_size
        return random_split(data, [train_size, test_size])
    if len(pct) == 3:
        train_size = round(len(data) * pct[0])
        val_size = round(len(data) * pct[1])
        test_size = len(data) - train_size - val_size
        return random_split(data, [train_size, val_size, test_size])


def select_top_k(data, top=2):
    arr = []
    for d in data:
        top_k_idx = d.argsort()[::-1][0:top]
        arr.append(d[top_k_idx])
    return np.array(arr)


if __name__ == '__main__':
    data_path = "../out/CIFAR10/GCN_CIFAR10_GPU0_13h39m49s_on_Sep_29_2020/"
    X_train_in, y_train_in, X_train_out, y_train_out = get_mem_data(data_path)
    get_selected_single_instances(X_train_in, X_train_in, 10)
