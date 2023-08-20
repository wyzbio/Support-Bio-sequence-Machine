# -*- encoding: utf-8 -*-
# @Time     :   2023/03/29 21:43:58
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   混合中心核依赖性最大化多核学习(Hybrid central kernel dependence maximization multi-kernel learning, HCKDM-MKL)

import numpy as np
import scipy
import scipy.io
from scipy.optimize import minimize, NonlinearConstraint
from _mini_tools import *
from _kernel_process import *

def k_nearest(X, k):
    n = X.shape[0]
    dis = np.zeros((n, n))
    sort_dis = np.zeros((n, n))
    ix = np.zeros((n, n), dtype=int)
    k_neighbors = np.zeros((n, k), dtype=int)
    
    for i in range(n):
        for j in range(i, n):
            dis[i, j] = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2))
            dis[j, i] = dis[i, j]

    for i in range(n):
        sort_dis[i, :], ix[i, :] = np.sort(dis[i, :]), np.argsort(dis[i, :])
        k_neighbors[i, :] = ix[i, 1:k+1]
    
    return k_neighbors


def kernel_matrix_centralization(K):
    """
    Function
    :param K: 
    :return: 
    """
    K_row_mean = np.mean(K, axis=1)
    K_col_mean = np.mean(K, axis=0)
    K_global_mean = np.mean(K)
    n = K.shape[0]
    K_centered = K - K_row_mean.reshape(n, 1) - K_col_mean + K_global_mean
    return K_centered


def generate_centring_matrix(num):
    e = (np.ones((num, 1)))
    I = np.identity(num)
    H = I - (np.matmul(e, np.transpose(e))) / num
    return H

def object_function(HSIC_global, kernel_global_list, HSIC_local_list, kernel_local_list_list, L_K, k, lamda, nu_1, nu_2):
    """
    Generate the objective function according to the formula
    :param K: K contains P kernel matrices, -1/N^2 tr(KHUH), multiply it by the beta and get k_asterisk
    :param L_K: The graph Laplacian matrix L_K
    :param nu_1: The graph regularization term
    :param nu_2: The L2 norm regularization term
    :return: The objective function
    """
    def function(beta):
        beta_rs = beta.reshape((20, 1, 1))

        # k = 100
        # lamda = 0.5
        # nu_1 = 1e-30
        # nu_2 = 1e-30

        # global
        weighted_kernel_matrix = kernel_global_list * beta_rs
        fused_kernel = np.sum(weighted_kernel_matrix, axis=0)
        global_term = np.trace(np.matmul(HSIC_global, fused_kernel))
        # local
        local_term = 0
        for i in range(kernel_local_list_list.shape[0]):
            weighted_kernel_matrix = kernel_local_list_list[i] * beta_rs
            fused_kernel = np.sum(weighted_kernel_matrix, axis=0)        
            local = np.trace(np.matmul(HSIC_local_list[i], fused_kernel))
            local_term = local_term + local

        term1 = -global_term * (1 - lamda) * (1 / kernel_local_list_list.shape[0])
        # term2 = -local_term * lamda * (1 / kernel_local_list_list.shape[0])
        term2 = -local_term * lamda * (1 / kernel_local_list_list.shape[0]) * (1 / (k * k))
        term3 = nu_1 * np.matmul(np.matmul(beta, L_K), beta)
        term4 = nu_2 * (np.linalg.norm(beta, 2) ** 2)
        # print('term1', term1)
        # print('term2', term2)
        # print('term3', term3)
        # print('term4', term4)
        # print('weight', beta)
        return (term1 + term2 + term3 + term4) * 1e30

    return function
    


def HCKDM(kernel_list, Y_train, k, lamda, nu1, nu2):
    """
    Using Hilbert Schmidt independence criterion to compute the weight of kernel matrices
    :param kernel_list: Contains all the kernel matrices, [kernel_number, kernel_rows, kernel_column]
    :param Y_train: link adjacent matrix F*
    :param nu_1: graph regularization term
    :param nu_2: L2 norm regularization term
    :return: the weight of kernel matrices
    """

    # ------------ Centralization ------------#
    for i in range(kernel_list.shape[0]):
        kernel_list[i] = kernel_matrix_centralization(kernel_list[i])

    #------------ Variate ------------#
    N = kernel_list.shape[1]
    P = kernel_list.shape[0]

    if Y_train.shape[0] == N:
        U = np.matmul(Y_train, np.transpose(Y_train))
    else:
        U = np.matmul(np.transpose(Y_train), Y_train)   
    
    H_global = generate_centring_matrix(N)
    H_local = generate_centring_matrix(k)

    
    #------------ Global ------------#
    HSIC_global = np.matmul(H_global, U)
    HSIC_global = np.matmul(HSIC_global, H_global)
    kernel_global_list = kernel_list

    #------------ Local ------------#
    mean_kernel = np.mean(kernel_list, axis=0)
    yita_matrix = k_nearest(mean_kernel, k)
    HSIC_local_list = []
    for i in range(N):
        U_local = U[np.ix_(yita_matrix[i], yita_matrix[i])]
        HSIC_local = np.matmul(H_local, U_local)
        HSIC_local = np.matmul(HSIC_local, H_local)
        HSIC_local_list.append(HSIC_local)
    HSIC_local_list = np.array(HSIC_local_list)
    # print('HSIC_local_list.shape', HSIC_local_list.shape)

    #------------ Local ------------#

    kernel_local_list_list = []
    for i in range(N):
        kernel_local_list = []
        for j in range(P):
            kernel_local = kernel_list[j][np.ix_(yita_matrix[i], yita_matrix[i])]
            kernel_local_list.append(kernel_local)
        kernel_local_list_list.append(kernel_local_list)
    kernel_local_list_list = np.array(kernel_local_list_list)
    # print('kernel_local_list_list.shape', kernel_local_list_list.shape)
       


    #------------ Regularization term ------------#
    W = np.zeros((P, P))
    for i in range(P):
        for j in range(i+1):
            A = kernel_list[i]
            B = kernel_list[j]
            inner_product = np.sum(A * B)
            norm_A = np.sqrt(np.sum(A ** 2))
            norm_B = np.sqrt(np.sum(B ** 2))
            W[i, j] = inner_product / (norm_A * norm_B)
            W[j, i] = W[i, j]
    W_rows_sum = np.sum(W, axis=0)
    D = np.diag(W_rows_sum)
    L_K = D - W

    np.random.seed(0)
    beta = np.random.rand(P)
    fun = object_function(HSIC_global, kernel_global_list, HSIC_local_list, kernel_local_list_list, L_K, k, lamda, nu1, nu2)
    # fun(beta)
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bnd = tuple((0, 1) for x in np.ones(P))
    res = minimize(fun, beta, method='SLSQP', bounds=bnd, constraints=cons)
    beta = res.x

    return beta


if __name__ == "__main__":
    #-------- Hyperparameter Setting Area--------#
    task_index_list = range(1, 2)
    dict_index_list = range(10)
    k_list = [50]
    lamda_list = [0.5]
    nu1_list = [1e-30]
    nu2_list = [1e-30]
    alpha_list = [-2 ** i for i in range(-4, 5)]
    c_list = [2 ** i for i in range(-4, 7)]
    para_tuple_list = [(a,b,c,d) for a in k_list for b in lamda_list for c in nu1_list for d in nu2_list]
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

        for param_com, (k, lamda, nu1, nu2) in enumerate(para_tuple_list):

            weight = HCKDM(train_kernel_list, y_train, k, lamda, nu1, nu2)
            # print(k, lamda, nu1, nu2)
            print(weight)