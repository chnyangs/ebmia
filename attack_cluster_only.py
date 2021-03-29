import numpy as np
from utils.DataUtil import get_mem_data, kmeans_generation
from sklearn import decomposition

if __name__ == '__main__':
    # 1. Load domain 1 dataset
    domain_1_path = "results/Target_DD_GCN_1616387754"
    domain_2_path = "out/CIFAR10/GCN_CIFAR10_GPU0_20h26m05s_on_Sep_28_2020"
    X_train_in_as_non_member, y_train_in_as_non_member, \
    X_train_out_as_non_member, y_train_out_as_non_member = get_mem_data(domain_1_path)

    X_train_in_as_target, y_train_in_as_target, \
    X_train_out_as_target, y_train_out_as_target = get_mem_data(domain_2_path)
    # prepare non-member dataset
    X_non_member = np.concatenate((X_train_in_as_non_member, X_train_out_as_non_member), axis=0)

    # prepare target dataset to evaluate
    X_target = np.concatenate((X_train_in_as_target, X_train_out_as_target), axis=0)
    # pca = decomposition.PCA(n_components=2)
    # pca.fit(X_target)
    # X_target = pca.transform(X_target)
    print(X_train_in_as_target.shape, X_train_out_as_target.shape, X_target.shape)

    # X_all = np.concatenate((X_target, X_non_member), axis=0)
    print("X_non_member:{}".format(X_non_member.shape))
    # print("X_all:{}".format(X_all.shape))
    label_idx = kmeans_generation(X_target, 2)
    member = label_idx[0]
    non_member = label_idx[1]
    breakpoint()