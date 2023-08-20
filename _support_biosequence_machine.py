# -*- encoding: utf-8 -*-
# @Time     :   2023/03/27 14:38:47
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   None

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from _kernel_process import kernel_parametere
from _mini_tools import *

# @timer
def SBM(train_kernel, test_kernel, train_label, test_label, alpha_list, c_list):
    result_list = []
    for alpha in alpha_list:
        parametered_train_kernel = kernel_parametere(alpha, train_kernel)
        parametered_test_kernel = kernel_parametere(alpha, test_kernel)
        for c in c_list:
            # print(c)
            SBM = svm.SVC(C=c, kernel='precomputed')
            SBM.fit(parametered_train_kernel, train_label)
            test_predict = SBM.predict(parametered_test_kernel)
            acc = accuracy_score(test_label, test_predict)
            result_list.append([alpha, c, acc])
    result = np.array(result_list)
    return np.max(result[:, 2]), result

def SBM_CV(train_kernel_fold_list, test_kernel_fold_list, train_label_fold_list, test_label_fold_list, alpha_list, c_list):
    result_list = []
    for alpha in alpha_list:
        print('a---')
        for c in c_list:
            print('c---')
            acc_all_fold = 0
            for fold in range(len(train_kernel_fold_list)):
                parametered_train_kernel = kernel_parametere(alpha, train_kernel_fold_list[fold])
                parametered_test_kernel = kernel_parametere(alpha, test_kernel_fold_list[fold])
                SBM = svm.SVC(C=c, kernel='precomputed')
                SBM.fit(parametered_train_kernel, train_label_fold_list[fold])
                test_predict = SBM.predict(parametered_test_kernel)
                acc = accuracy_score(test_label_fold_list[fold], test_predict)
                acc_all_fold = acc_all_fold + acc
            result_list.append([alpha, c, acc_all_fold/len(train_kernel_fold_list)])
    result = np.array(result_list)
    return np.max(result[:, 2]), result


if __name__ == "__main__":
    
    pass