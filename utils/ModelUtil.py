from functools import partial
import numpy as np
from utils.MIUtil import compute_pairwise_distances
import tensorflow as tf
from tqdm import tqdm

def gaussian_kernel_matrix(x, y, sigmas):
    r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
    We create a sum of multiple gaussian kernels each having a width sigma_i.
    Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      sigmas: a tensor of floats which denote the widths of each of the
        gaussians in the kernel.
    Returns:
      A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
    """
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

    dist = compute_pairwise_distances(x, y)

    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    '''
    Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
    the distributions of x and y. Here we use the kernel two sample estimate
    using the empirical mean of the two distributions.
    MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
                = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },
    where K = <\phi(x), \phi(y)>,
      is the desired kernel function, in this case a radial basis kernel.
    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        kernel: a function which computes the kernel in MMD. Defaults to the
                GaussianKernelMatrix.
    Returns:
        a scalar denoting the squared maximum mean discrepancy loss.
    '''
    with tf.name_scope('MaximumMeanDiscrepancy'):
        # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
        cost = tf.reduce_mean(kernel(x, x))
        cost += tf.reduce_mean(kernel(y, y))
        cost -= 2 * tf.reduce_mean(kernel(x, y))

        # We do not allow the loss to become negative.
        cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
    """Adds a similarity loss term, the MMD between two representations.
    This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
    different Gaussian kernels.
    Args:
      source_samples: a tensor of shape [num_samples, num_features].
      target_samples: a tensor of shape [num_samples, num_features].
      weight: the weight of the MMD loss.
      scope: optional name scope for summary tags.
    Returns:
      a scalar tensor representing the MMD loss value.
    """
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight

    return loss_value


def evaluate_attack(m_true, m_pred):
    accuracy = tf.keras.metrics.Accuracy()
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()
    accuracy.update_state(m_true, m_pred)
    precision.update_state(m_true, m_pred)
    recall.update_state(m_true, m_pred)
    F1_Score = 2 * (precision.result() * recall.result()) / (precision.result() + recall.result())
    print('accuracy:%.4f precision:%.4f recall:%.4f F1_Score:%.4f'
          % (accuracy.result(), precision.result(), recall.result(), F1_Score))


def evaluate_cluster_distance_attack(params):
    X_target, target_number, selected_data, max_original_dist = params
    data_size = X_target.shape[0]
    non_member_correct = 0
    member_correct = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    for moves_index in tqdm(range(data_size),desc="evaluate_cluster_distance_attack:"):
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
            if max_original_dist >= moves_target_to_non_member_dist:
                member_correct += 1
                TP += 1
            else:
                # member -> non-member
                FN += 1
        else:
            # moves non-members
            if max_original_dist <= moves_target_to_non_member_dist:
                non_member_correct += 1
                TN += 1
            else:
                # non-member -> member
                FP += 1
    # calculate accuracy, precision, recall and F1-Score
    accuracy = (non_member_correct + member_correct) / data_size
    precision = TP / (TP + FP) # low precision, high FP
    recall = TP / (TP + FN) # high recall, low FN
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1