# -*- coding: utf-8 -*-
# @Time    : 2021/12/13 10:31
# @Author  : Yizheng Wang
# @E-mail  : wyz020@126.com
# @File    : HSIC.py

import numpy as np
import scipy
import scipy.io
from scipy.optimize import minimize, NonlinearConstraint

def object_function(K, L_K, nu_1, nu_2):
    """
    Generate the objective function according to the formula
    :param K:K contains P kernel matrices, -1/N^2 tr(KHUH), multiply it by the beta and get k_asterisk
    :param L_K:The graph Laplacian matrix L_K
    :param nu_1:The graph regularization term
    :param nu_2:The L2 norm regularization term
    :return:The objective function
    """
    f = lambda beta: -np.matmul(np.transpose(beta), K) \
                     + nu_1 * np.matmul(np.matmul(np.transpose(beta), L_K), beta) \
                     + nu_2 * (np.linalg.norm(beta, 2) ** 2)
    return f


def HSIC(kernel_list, Y_train, nu_1=0.1, nu_2=0.01):
    """
    Using Hilbert Schmidt independence criterion to compute the weight of kernel matrices
    :param kernel_list:Contains all the kernel matrices, [kernel_number, kernel_rows, kernel_column]
    :param Y_train:link adjacent matrix F*
    :param nu_1:graph regularization term
    :param nu_2:L2 norm regularization term
    :return:the weight of kernel matrices
    """
    # Define some fundamental quantities
    N = kernel_list.shape[1]  # The number of rows of the kernel matrix
    P = kernel_list.shape[0]  # The number of kernel matrices
    e = (np.ones((N, 1)))
    I = np.identity(N)
    H = I - (np.matmul(e, np.transpose(e))) / N

    # # Compute graph Laplacian matrix L_K
    # W = np.ones((P, P))
    # for i in range(P):
    #     for j in range(P):
    #         A = kernel_list[i]
    #         A_T = np.transpose(A)
    #         B = kernel_list[j]
    #         B_T = np.transpose(B)
    #         Frobenius_inner_product = np.trace(np.matmul(A_T, B))
    #         Frobenius_norm_A = np.trace(np.matmul(A_T, A)) ** 0.5
    #         Frobenius_norm_B = np.trace(np.matmul(B_T, B)) ** 0.5
    #         W[i][j] = Frobenius_inner_product / (Frobenius_norm_A * Frobenius_norm_B)
    
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

    # Obtain objective function
    if Y_train.shape[0] == N:
        U = np.matmul(Y_train, np.transpose(Y_train))
    else:
        U = np.matmul(np.transpose(Y_train), Y_train)

    K = np.ones((P, 1))
    for i in range(P):
        HSIC_item = np.matmul(kernel_list[i], H)
        HSIC_item = np.matmul(HSIC_item, U)
        HSIC_item = np.trace(np.matmul(HSIC_item, H))
        K[i] = HSIC_item / (N ** 2)
        

    fun = object_function(K, L_K, nu_1, nu_2)

    np.random.seed(0)
    beta = np.random.rand(P, 1)

    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    bnd = tuple((0, 1) for x in np.ones(P))  

    res = minimize(fun, beta, method='SLSQP', bounds=bnd, constraints=cons)
    beta = res.x

    return beta


if __name__ == "__main__":
    
   pass