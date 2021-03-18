import os
import numpy as np
import tensorflow as tf
from utils.DataUtil import get_all_probs_path, get_mem_data
from utils.ModelUtil import mmd_loss

if __name__ == '__main__':

    path_infos = get_all_probs_path('out')
    # , 'DD', 'ENZYMES', 'CIFAR10', 'MNIST' ,'OGBG',
    dataset_list = ['OGBG']
    for selected_dataset in dataset_list:
        # for exp in path_infos[selected_dataset]:
        dataset_path = os.path.join('out', selected_dataset)
        exps = path_infos[selected_dataset]

        for exp in exps:
            m_pred, m_true = [], []
            exp_path = os.path.join(dataset_path, exp)
            X_train_in, y_train_in, X_train_out, y_train_out = get_mem_data(exp_path)
            print(X_train_in.shape, X_train_out.shape)
            moves_distance_in = 0
            non_member_correct = 0
            member_correct = 0
            wrong = 0
            nn,mm,nm,mn = 0,0,0,0

            data_size = min(X_train_in.shape[0], X_train_out.shape[0])
            # if X_train_in.shape[0] != X_train_out.shape[0]:
            #     continue
            for moves_index in range(data_size):
                tmp_out = X_train_out[moves_index]
                moves_X_train_in = np.append(X_train_in,[tmp_out],axis=0)
                tmp_in = X_train_in[moves_index]
                moves_X_train_out = np.append(X_train_out, [tmp_in], axis=0)

                moves_X_train_in = tf.convert_to_tensor(moves_X_train_in, dtype=float)
                moves_X_train_out = tf.convert_to_tensor(moves_X_train_out, dtype=float)
                X_train_in = tf.convert_to_tensor(X_train_in, dtype=float)
                X_train_out = tf.convert_to_tensor(X_train_out, dtype=float)
                ori_dist = mmd_loss(X_train_in, X_train_out, 1)
                moves_non2mem_dist = mmd_loss(moves_X_train_in, X_train_out, 1)
                moves_mem2non_dist = mmd_loss(X_train_in, moves_X_train_out, 1)
                # print("Index:{}\n\t".format(moves_index))
                # print("Origional:{}".format(ori_dist))
                # print("Move non-member to member:{}".format(moves_non2mem_dist))
                # print("Move member to non-member:{}".format(moves_mem2non_dist))
                if ori_dist <= moves_non2mem_dist:
                    non_member_correct += 1
                elif moves_mem2non_dist <= ori_dist:
                    member_correct += 1
                else:
                    wrong += 1
            print("dataset:{}".format(selected_dataset))
            print("experiment path:{}".format(exp_path))
            print(non_member_correct / data_size)
            print(member_correct / data_size)
            print(wrong / data_size)

