# -*- encoding: utf-8 -*-
# @Time     :   2023/03/11 15:27:09
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   序列核生成，独立验证集，单核，Levenshtein距离

from multiprocessing import Pool
import numpy as np
from _mini_tools import read_fasta
from _calculate_sw_similarity import compute_smithwaterman_train_kernel, compute_smithwaterman_test_kernel
from _dictionaries_for_grouping import dictionaries, replace
from _kernel_process import max_sequence_len_normalize_train, max_sequence_len_normalize_test


if __name__ == "__main__":

    #-------- Hyperparameter Setting Area--------#
    task_index_list = range(1, 7)
    process_num = 80
    tag = 'sw'
    #--------------------------------------------#

    for task_index in task_index_list:
        print('Task %s is running'%task_index)

        #------------ sample ------------#
        sample_train_filename = 'datasets/%s/%s_train_sample.fasta'%(task_index, task_index)
        sample_test_filename = 'datasets/%s/%s_test_sample.fasta'%(task_index, task_index)
        sample_train = read_fasta(sample_train_filename)
        sample_test = read_fasta(sample_test_filename)

        #------------ similarity ------------#
        train_kernel = np.array(compute_smithwaterman_train_kernel(sample_train, process_num))
        test_kernel = np.array(compute_smithwaterman_test_kernel(sample_test, sample_train, process_num))

        #------------ normalize ------------#
        normalized_train_kernel = max_sequence_len_normalize_train(train_kernel, sample_train)
        normalized_test_kernel = max_sequence_len_normalize_test(test_kernel, sample_train, sample_test)

        #------------ save kernel ------------#
        train_kernel_save_filename = 'intermediate_results/%s/%s_train_kernel_ug_%s.npy'%(task_index, task_index, tag)
        test_kernel_save_filename = 'intermediate_results/%s/%s_test_kernel_ug_%s.npy'%(task_index, task_index, tag)
        np.save(train_kernel_save_filename, normalized_train_kernel)
        np.save(test_kernel_save_filename, normalized_test_kernel)