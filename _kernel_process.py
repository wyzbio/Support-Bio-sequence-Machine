# -*- encoding: utf-8 -*-
# @Time     :   2023/03/05 11:49:54
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   保存核处理方法，包括[线性归一化，剪裁归一化+经验准则]两种核归一化方法、核参数化处理方法


import numpy as np

def linear_normalize(a):
    m = a.max()
    return (m - a)/m


def clipping_normalize(sigma, k):
    k_flatten = k.flatten()
    print(k_flatten.shape)
    means_value = k_flatten.mean()
    std_value = k_flatten.std()
    lower_bound = means_value - sigma * std_value
    upper_bound = means_value + sigma * std_value
    r1 = np.where(k<upper_bound, k, upper_bound)
    r2 = np.where(r1>lower_bound, r1, lower_bound)
    return (upper_bound - r2)/upper_bound


def kernel_parametere(alpha, k):
    r = np.array(k, dtype=np.float64)
    max_every_rows = np.max(r, axis=1).reshape(-1,1)
    max_mat = np.repeat(max_every_rows, repeats=r.shape[1], axis=1)
    return (1 - np.exp(alpha * (k - max_mat) / max_mat))


def max_sequence_len_normalize_train(train_ker, train_sam):
    sequence_len_list = [len(train_sam[i]) for i in range(len(train_sam))]
    result = [[max(sequence_len_list[i], sequence_len_list[j]) for j in range(len(sequence_len_list))] for i in range(len(sequence_len_list))]
    train_ker = train_ker/result
    return train_ker


def max_sequence_len_normalize_test(test_ker, train_sam, test_sam):
    train_len_list = [len(train_sam[i]) for i in range(len(train_sam))]
    test_len_list = [len(test_sam[i]) for i in range(len(test_sam))]
    result = [[max(test_len_list[i], train_len_list[j]) for j in range(len(train_len_list))] for i in range(len(test_len_list))]
    test_ker = test_ker/result
    return test_ker


def extract_kernel(kernel, train_index_list, test_index_list):
    # 提取对称矩阵m1中的特定行和列
    train_kernel = kernel[np.ix_(train_index_list, train_index_list)]
    test_kernel = kernel[np.ix_(test_index_list, train_index_list)]
    return train_kernel, test_kernel


if __name__ == '__main__':
    pass