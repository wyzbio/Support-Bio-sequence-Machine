# -*- encoding: utf-8 -*-
# @Time     :   2023/03/27 10:12:02
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   支持生物序列机，独立验证集，单核，Levenshtein距离


import numpy as np
from _kernel_process import max_sequence_len_normalize_train, max_sequence_len_normalize_test, linear_normalize
from _mini_tools import *
from _support_biosequence_machine import SBM


if __name__ == "__main__":

    #-------- Hyperparameter Setting Area--------#
    task_index_list = range(1, 7)
    alpha_list = [-2 ** i for i in range(-4, 5)]
    c_list = [2 ** i for i in range(-10, 11)]
    tag = 'ls'
    #--------------------------------------------#

    for task_index in task_index_list:

        print('Task %s is running'%task_index)

        #------------ label ------------#
        train_label_filename = 'datasets/%s/%s_train_label.txt'%(str(task_index), str(task_index))
        test_label_filename = 'datasets/%s/%s_test_label.txt'%(str(task_index), str(task_index))
        train_label = np.array(read_txt(train_label_filename)).flatten()
        test_label = np.array(read_txt(test_label_filename)).flatten()
    
        #------------ kernel ------------#
        train_kernel_filename = 'intermediate_results/%s/%s_train_kernel_ug_%s.npy'%(str(task_index), str(task_index), tag)
        test_kernel_filename = 'intermediate_results/%s/%s_test_kernel_ug_%s.npy'%(str(task_index), str(task_index), tag)
        train_kernel = np.load(train_kernel_filename)
        test_kernel = np.load(test_kernel_filename)

        max_result, result = SBM(train_kernel, test_kernel, train_label, test_label, alpha_list, c_list)
        print(max_result)
        np.set_printoptions(suppress=True)
        save_filename = 'experimental_results/%s/%s_result_ug_%s.txt'%(task_index, task_index, tag)
        save_list_to_txt(save_filename, result)
            