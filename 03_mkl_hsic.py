# -*- encoding: utf-8 -*-
# @Time     :   2023/04/23 23:52:42
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   None


import numpy as np
import scipy
import scipy.io
from scipy.optimize import minimize, NonlinearConstraint
from _mini_tools import *
from _kernel_process import *
from _MKL_HCKDM import HCKDM
from _MKL_HSIC import HSIC


if __name__ == "__main__":

    #-------- Hyperparameter Setting Area--------#
    task_index_list = range(1, 7)
    dict_index_list = range(10)
    #--------------------------------------------#
    
    for task_index in task_index_list:

        print('Task %s is running'%task_index)

        #------------ label ------------#
        train_label_filename = 'datasets/%s/%s_train_label.txt'%(task_index, task_index)
        train_label = read_row_txt((train_label_filename))

        #------------ kernel ------------#
        train_kernel_list = []
        test_kernel_list = []

        for dict_index in dict_index_list:
            train_kernel_filename = 'intermediate_results/%s/%s_train_kernel_d%s_ls.npy'%(task_index, task_index, dict_index)
            test_kernel_filename = 'intermediate_results/%s/%s_test_kernel_d%s_ls.npy'%(task_index, task_index, dict_index)
            train_kernel_list.append(np.load(train_kernel_filename))
            test_kernel_list.append(np.load(test_kernel_filename))
            
        for dict_index in dict_index_list:
            train_kernel_filename = 'intermediate_results/%s/%s_train_kernel_d%s_sw.npy'%(task_index, task_index, str(dict_index))
            test_kernel_filename = 'intermediate_results/%s/%s_test_kernel_d%s_sw.npy'%(task_index, task_index, str(dict_index))
            train_kernel_list.append(np.load(train_kernel_filename))
            test_kernel_list.append(np.load(test_kernel_filename))

        train_kernel_list = np.array(train_kernel_list)
        test_kernel_list = np.array(test_kernel_list)

        y_train_list = [[1, 0] if x == 1 else [0, 1] for x in train_label]
        y_train = np.array(y_train_list)

        weight = HSIC(train_kernel_list, y_train)
        weight = np.array(weight).reshape(-1,1)
        weight = weight.reshape((train_kernel_list.shape[0], 1, 1))
               
        train_weight_broadcast = np.broadcast_to(weight, (train_kernel_list.shape[0], train_kernel_list.shape[1], train_kernel_list.shape[2]))
        test_weight_broadcast = np.broadcast_to(weight, (train_kernel_list.shape[0], test_kernel_list.shape[1], test_kernel_list.shape[2]))

        fused_train_kernel = np.sum(train_kernel_list * train_weight_broadcast, axis=0)
        fused_test_kernel = np.sum(test_kernel_list * test_weight_broadcast, axis=0)

        train_kernel_save_filename = 'intermediate_results/%s/%s_train_kernel_hsic.npy'%(task_index, task_index)
        test_kernel_save_filename = 'intermediate_results/%s/%s_test_kernel_hsic.npy'%(task_index, task_index)
        print(fused_train_kernel)
        np.save(train_kernel_save_filename, fused_train_kernel)
        np.save(test_kernel_save_filename, fused_test_kernel)
