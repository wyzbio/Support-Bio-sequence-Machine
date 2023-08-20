# -*- encoding: utf-8 -*-
# @Time     :   2023/03/11 15:27:09
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   序列核生成，独立验证集，单核，Levenshtein距离

from multiprocessing import Pool
import numpy as np
from _mini_tools import read_fasta
from _calculate_ls_similarity import compute_levenshetein_train_kernel, compute_levenshetein_test_kernel
from _dictionaries_for_grouping import dictionaries, replace
from _kernel_process import linear_normalize


if __name__ == "__main__":
      
    #-------- Hyperparameter Setting Area--------#
    task_index_list = range(1, 7)
    process_num = 40
    tag = 'ls'
    #--------------------------------------------#
    
    for task_index in task_index_list:
        print('Task %s is running'%task_index)

        #------------ sample ------------#
        sample_train_filename = 'datasets/%s/%s_train_sample.fasta'%(task_index, task_index)
        sample_test_filename = 'datasets/%s/%s_test_sample.fasta'%(task_index, task_index)
        sample_train = read_fasta(sample_train_filename)
        sample_test = read_fasta(sample_test_filename)

        #------------ similarity ------------#
        train_kernel = np.array(compute_levenshetein_train_kernel(sample_train, process_num))
        test_kernel = np.array(compute_levenshetein_test_kernel(sample_test, sample_train, process_num))

        #------------ normalize ------------#
        normalized_train_kernel = linear_normalize(train_kernel)
        normalized_test_kernel = linear_normalize(test_kernel)

        #------------ save kernel ------------#
        train_kernel_save_filename = 'intermediate_results/%s/%s_train_kernel_ug_%s.npy'%(task_index, task_index, tag)
        test_kernel_save_filename = 'intermediate_results/%s/%s_test_kernel_ug_%s.npy'%(task_index, task_index, tag)
        np.save(train_kernel_save_filename, normalized_train_kernel)
        np.save(test_kernel_save_filename, normalized_test_kernel)