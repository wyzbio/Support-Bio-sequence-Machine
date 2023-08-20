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

        for dict_index in range(len(dictionaries)):
            print('Dictionary %s is running'%dict_index)

            #------------ sample ------------#
            sample_train_filename = 'datasets/%s/%s_train_sample.fasta'%(task_index, task_index)
            sample_test_filename = 'datasets/%s/%s_test_sample.fasta'%(task_index, task_index)
            sample_train = read_fasta(sample_train_filename)
            sample_test = read_fasta(sample_test_filename)

            #------------ replace ------------#
            replaced_sample_train = replace(sample_train, dictionaries[dict_index])
            replaced_sample_test = replace(sample_test, dictionaries[dict_index])

            #------------ similarity ------------#
            train_kernel = np.array(compute_levenshetein_train_kernel(replaced_sample_train, process_num))
            test_kernel = np.array(compute_levenshetein_test_kernel(replaced_sample_test, replaced_sample_train, process_num))

            #------------ normalize ------------#
            normalized_train_kernel = linear_normalize(train_kernel)
            normalized_test_kernel = linear_normalize(test_kernel)

            #------------ save kernel ------------#
            train_kernel_save_filename = 'intermediate_results/%s/%s_train_kernel_d%s_%s.npy'%(task_index, task_index, dict_index, tag)
            test_kernel_save_filename = 'intermediate_results/%s/%s_test_kernel_d%s_%s.npy'%(task_index, task_index, dict_index, tag)
            np.save(train_kernel_save_filename, normalized_train_kernel)
            np.save(test_kernel_save_filename, normalized_test_kernel)