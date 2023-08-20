# -*- encoding: utf-8 -*-
# @Time     :   2023/03/27 10:12:02
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   支持生物序列机，独立验证集，单核，Levenshtein距离


import numpy as np
from _kernel_process import max_sequence_len_normalize_train, max_sequence_len_normalize_test, linear_normalize
from _mini_tools import *
from _support_biosequence_machine import SBM_CV


if __name__ == "__main__":

    #-------- Hyperparameter Setting Area--------#
    task_index_list = range(7, 11)
    dict_index_list = range(10)
    fold_index_list = range(10)
    alpha_list = [-2 ** i for i in range(-4, 5)]
    c_list = [2 ** i for i in range(-10, 11)]
    tag = 'sw'
    #--------------------------------------------#

    for task_index in task_index_list:

        print('Task %s is running'%task_index)

        for dict_index in dict_index_list:

            train_kernel_fold_list = []
            test_kernel_fold_list = []
            train_label_fold_list = []
            test_label_fold_list = []

            for fold_index in fold_index_list:

                #------------ label ------------#
                train_label_filename = 'datasets/%s/%s_train_label_cv%s.txt'%(str(task_index), str(task_index), fold_index)
                test_label_filename = 'datasets/%s/%s_test_label_cv%s.txt'%(str(task_index), str(task_index), fold_index)
                train_label = np.array(read_txt(train_label_filename)).flatten()
                test_label = np.array(read_txt(test_label_filename)).flatten()
                train_label_fold_list.append(train_label)
                test_label_fold_list.append(test_label)
            
                #------------ kernel ------------#
                train_kernel_filename = 'intermediate_results/%s/%s_train_kernel_d%s_cv%s_%s.npy'%(str(task_index), str(task_index), str(dict_index), fold_index, tag)
                test_kernel_filename = 'intermediate_results/%s/%s_test_kernel_d%s_cv%s_%s.npy'%(str(task_index), str(task_index), str(dict_index), fold_index, tag)
                train_kernel = np.load(train_kernel_filename)
                test_kernel = np.load(test_kernel_filename)
                train_kernel_fold_list.append(train_kernel)
                test_kernel_fold_list.append(test_kernel)

            max_result, result = SBM_CV(train_kernel_fold_list, test_kernel_fold_list, train_label_fold_list, test_label_fold_list, alpha_list, c_list)
            print(max_result)
            np.set_printoptions(suppress=True)
            save_filename = 'experimental_results/%s/%s_result_d%s_%s.txt'%(task_index, task_index, dict_index, tag)
            save_list_to_txt(save_filename,result)
            
                
