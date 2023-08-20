# -*- encoding: utf-8 -*-
# @Time     :   2023/03/29 21:43:58
# @Author   :   Yizheng Wang
# @E-mail   :   wyz020@126.com
# @Function :   None


import numpy as np
from scipy.optimize import minimize
import numpy as np


def k_nearest(X, k):
    n = X.shape[0]
    dis = np.zeros((n, n))
    sort_dis = np.zeros((n, n))
    ix = np.zeros((n, n), dtype=int)
    k_neighbors = np.zeros((n, k), dtype=int)
    
    # compute distance
    for i in range(n):
        for j in range(i, n):
            dis[i, j] = np.sqrt(np.sum((X[i, :] - X[j, :]) ** 2))
            dis[j, i] = dis[i, j]
    
    # sort and select k neighbors
    for i in range(n):
        sort_dis[i, :], ix[i, :] = np.sort(dis[i, :]), np.argsort(dis[i, :])
        k_neighbors[i, :] = ix[i, 1:k+1]
    
    return k_neighbors


def k_local(K_C, K_y, k):
    k_near = k_nearest(K_C, k)
    Klocal_C = np.zeros((K_y.shape[0], k, k, K_C.shape[-1]))
    Klocal_y = np.zeros((K_y.shape[0], k, k))
    
    for v in range(K_y.shape[0]):
        for i in range(k):
            for j in range(k):
                Klocal_C[v, i, j, :] = K_C[k_near[v, i], k_near[v, j], :]
                Klocal_y[v, i, j] = K_y[k_near[v, i], k_near[v, j]]
    
    return Klocal_C, Klocal_y


def HKAM(Kernels_list, adjmat, k=10, lamda=0.01):
    Kernels_list = np.transpose(Kernels_list, (1, 2, 0))  
    num_samples = Kernels_list.shape[0]
    num_kernels = Kernels_list.shape[2]

    y = adjmat
    ga = np.matmul(y, y.T)

    Kglo_C, Kglo_y = global_kernel(Kernels_list, ga)
    
    Klocal_C, Klocal_y = k_local(Kglo_C, Kglo_y, k)
    N_U = ga.shape[0]
    l = np.ones((N_U, 1))
    H = np.eye(N_U) - np.matmul(l, l.T) / N_U
    # print(Kglo_C.shape, Kglo_y.shape, l.shape, H.shape)

    # M = np.zeros((num_kernels, num_kernels))
    # for i in range(num_kernels):
    #     for j in range(num_kernels):
    #         kk1 = np.matmul(np.matmul(H, Kglo_C[:, :, i]), H)
    #         kk2 = np.matmul(np.matmul(H, Kglo_C[:, :, j]), H)
    #         mm = np.trace(np.matmul(kk1.T, kk2))
    #         m1 = np.trace(np.matmul(kk1, kk1.T))
    #         m2 = np.trace(np.matmul(kk2, kk2.T))
    #         M[i, j] = mm / (np.sqrt(m1) * np.sqrt(m2))

    kk = np.zeros((num_kernels, num_samples, num_samples))
    for i in range(num_kernels):
        print(i)
        kk[i] = np.matmul(np.matmul(H, Kglo_C[:, :, i]), H)
        
    M = np.zeros((num_kernels, num_kernels))
    for i in range(num_kernels):
        for j in range(num_kernels):
            print(j)
            kk1 = kk[i]
            kk2 = kk[j]
            mm = np.einsum('ij,ij->', kk1, kk2)
            m1 = np.einsum('ij,ij->', kk1, kk1)
            m2 = np.einsum('ij,ij->', kk2, kk2)
            M[i, j] = mm / (np.sqrt(m1) * np.sqrt(m2))

    M_local = np.zeros((num_samples, num_kernels, num_kernels))
    for v in range(num_samples):
        for i in range(num_kernels):
            for j in range(num_kernels):
                kk1 = np.squeeze(Klocal_C[v, :, :, i])
                kk2 = np.squeeze(Klocal_C[v, :, :, j])
                mm = np.trace(np.matmul(kk1.T, kk2))
                M_local[v, i, j] = mm
    # print(M_local)

    tau = np.ones((num_samples, 1))
    gamma = np.ones((num_kernels, 1))

    Q = compute_Q(Klocal_C, Klocal_y, tau, num_kernels, num_samples)
    Z = compute_Z(Kglo_C, Kglo_y, num_kernels, N_U, H)

    w, tau, obj, Q, Z = iter_obj(M, M_local, Q, Z, gamma, lamda, Kglo_C, Kglo_y, Klocal_C, Klocal_y, tau, num_kernels, num_samples)
    obj_1 = obj

    while True:
        w, tau, obj, Q, Z = iter_obj(M, M_local, Q, Z, gamma, lamda, Kglo_C, Kglo_y, Klocal_C, Klocal_y, tau, num_kernels, num_samples)
        print(obj)
        print(w)
        if (obj_1 - obj) / obj <= 1e-4:
            break
        obj_1 = obj

    return w


def global_kernel(K, K_Y):
    K_y = K_Y
    num_kernels = K.shape[2]
    K_C = np.zeros((K.shape[0], K.shape[1], num_kernels))

    for v in range(num_kernels):
        n = K.shape[0]
        f_ij = np.sum(np.diag(K[:, :, v])) / (n ** 2)
        f_j = np.sum(K[:, :, v], axis=0) / n
        f_i = np.sum(K[:, :, v], axis=1) / n

        for i in range(n):
            for j in range(i, n):
                K_C[i, j, v] = K[i, j, v] - f_j[j] - f_i[i] + f_ij
                K_C[j, i, v] = K_C[i, j, v]

    return K_C, K_y


def obj_function(M, Q, Z, gamma, lamda):
    J = np.matmul(np.matmul(gamma.T, M), gamma) - 2 * (1 - lamda) * np.matmul(gamma.T, Q) - 2 * lamda * np.matmul(gamma.T, Z)
    return J


def iter_obj(M, M_local, Q, Z, gamma, lamda, Kernels_list, ga, local_Kernels_list, K_y, tau, num_kernels, num_samples):
    falpha = lambda gamma: obj_function(M, Q, Z, gamma, lamda)
    gamma, fval = optimize_weights(gamma, falpha)
    weight_v = gamma / np.linalg.norm(gamma, 2)
    w = weight_v
    obj = ((1 - lamda) * np.matmul(weight_v.T, Q) - lamda * np.matmul(weight_v.T, Z)) / (np.matmul(np.matmul(weight_v.T, M), weight_v) ** 0.5)
    tau = compute_tau(M, M_local, weight_v, num_samples)
    Q = compute_Q(local_Kernels_list, K_y, tau, num_kernels, num_samples)
    Z = compute_Z(Kernels_list, ga, num_kernels, ga.shape[0], np.eye(ga.shape[0]) - np.ones((ga.shape[0], ga.shape[0])) / ga.shape[0])
    
    return w, tau, obj, Q, Z

def compute_Q(local_Kernels_list, K_y, tau, num_kernels, num_samples):
    Q = np.zeros((num_kernels, 1))
    for i in range(num_kernels):
        temp = 0
        for j in range(num_samples):
            temp += np.trace(np.matmul(np.squeeze(local_Kernels_list[j, :, :, i]).T, np.squeeze(K_y[j, :, :])))/tau[j]
        Q[i] = temp / num_samples
    return Q


def compute_Z(Kernels_list, ga, num_kernels, N_U, H):
    Z = np.zeros((num_kernels, 1))
    for i in range(num_kernels):
        kk = np.matmul(np.matmul(H, Kernels_list[:, :, i]), H)
        bb = np.trace(np.matmul(kk.T, ga))
        Z[i] = Kernels_list.shape[0] * bb * ((N_U - 1) ** -2)
    return Z


# def compute_tau(M, M_local, weight_v, num_samples):
#     tau = np.zeros((num_samples, 1))
#     for i in range(num_samples):
#         tau[i] = (np.matmul(np.matmul(weight_v.T, np.squeeze(M_local[i, :, :]), weight_v), 1) ** 0.5) / (np.matmul(np.matmul(weight_v.T, M), weight_v) ** 0.5)
#     return tau


def compute_tau(M, M_local, weight_v, num_samples):
    tau = np.zeros((num_samples,1))
    for i in range(num_samples):
        wv_M_local = np.squeeze(M_local[i,:,:]) @ weight_v
        tau_numerator = weight_v.T @ wv_M_local
        tau_denominator = weight_v.T @ M @ weight_v
        tau[i] = np.sqrt(tau_numerator / tau_denominator)
    return tau


def optimize_weights(x0, fun):
    n = len(x0)
    Aineq = None
    bineq = None
    Aeq = None
    beq = None
    bounds = [(0, None) for i in range(n)]
    options = {'disp': False}
    res = minimize(fun, x0, bounds=bounds, method='L-BFGS-B', options=options)
    return res.x, res.fun





if __name__ == "__main__":
    pass