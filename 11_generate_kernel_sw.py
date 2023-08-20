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
from _kernel_process import max_sequence_len_normalize_train, max_sequence_len_normalize_test, extract_kernel

if __name__ == "__main__":
    
    #-------- Hyperparameter Setting Area--------#
    task_index_list = range(8, 11)
    fold_index_list = range(10)
    process_num = 80
    tag = 'sw'
    #--------------------------------------------#
    for task_index in task_index_list:
        print('Task %s is running'%task_index)

        for dict_index in range(len(dictionaries)):
            print('Dictionary %s is running'%dict_index)

            #------------ all sample ------------#
            sample_all_filename = 'datasets/%s/%s_sample.fasta'%(task_index, task_index)
            sample_all = read_fasta(sample_all_filename)
            
            #------------ replace ------------#
            replaced_all_kernel = replace(sample_all, dictionaries[dict_index])

            #------------ similarity ------------#
            all_kernel = np.array(compute_smithwaterman_train_kernel(replaced_all_kernel, process_num))
        
            for fold_index in fold_index_list:
                
                #------------ extract kernel ------------#
                train_index_list = np.load('datasets/%s/%s_train_index_cv%s.npy'%(task_index, task_index, fold_index))
                test_index_list = np.load('datasets/%s/%s_test_index_cv%s.npy'%(task_index, task_index, fold_index))

                train_kernel, test_kernel = extract_kernel(all_kernel, train_index_list, test_index_list)
                
                #------------ all sample ------------#
                sample_train = [sample_all[i] for i in train_index_list]
                sample_test = [sample_all[i] for i in test_index_list]

                #------------ normalize ------------#
                normalized_train_kernel = max_sequence_len_normalize_train(train_kernel, sample_train)
                normalized_test_kernel = max_sequence_len_normalize_test(test_kernel, sample_train, sample_test)

                #------------ save kernel ------------#
                train_kernel_save_filename = 'intermediate_results/%s/%s_train_kernel_d%s_cv%s_%s.npy'%(task_index, task_index, dict_index, fold_index, tag)
                test_kernel_save_filename = 'intermediate_results/%s/%s_test_kernel_d%s_cv%s_%s.npy'%(task_index, task_index, dict_index, fold_index, tag)
                np.save(train_kernel_save_filename, normalized_train_kernel)
                np.save(test_kernel_save_filename, normalized_test_kernel)