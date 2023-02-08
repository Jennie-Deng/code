def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]




import numpy as np
from scipy.sparse.linalg import eigs
def Eigen_Reweighting(X,order,coef):
    # X: original eigenvalues
    # order: order, -1 stands for infinity
    # coef: weights, decaying constant if order = -1
    # return: reweighted eigenvalues
    if order == -1:     # infinity
        assert len(coef) == 1, 'Eigen_Reweighting wrong.'
        coef = coef[0]
        assert np.max(np.absolute(X)) * coef < 1, 'Decaying constant too large.'
        X_H = np.divide(X, 1 - coef * X)
    else:
        assert len(coef) == order, 'Eigen_Reweighting wrong.'
        X_H = coef[0] * X
        X_temp = X
        for i in range(1,order):
            X_temp = np.multiply(X_temp,X)
            X_H += coef[i] * X_temp
    return X_H


def Eigen_TopL(A, d):
    # A: N x N symmetric sparse adjacency matrix
    # d: preset dimension
    # return: top-L eigen-decomposition of A containing at least d positive eigenvalues
    # assert np.all(A.T == A), 'The matrix is not symmetric!'
    L = d + 10
    lambd = np.array([0])
    while sum(lambd > 0) < d:         # can be improved to reduce redundant calculation if L <= 2d + 10 not hold
        L = L + d
        lambd, X = eigs(A, L)
        lambd, X = lambd.real, X.real
        # only select top-L
    temp_index = np.absolute(lambd).argsort()[::-1]
    lambd = lambd[temp_index]
    temp_max, = np.where(np.cumsum(lambd > 0) >= d)
    lambd, temp_index = lambd[:temp_max[0]+1], temp_index[:temp_max[0]+1]
    X = X[:,temp_index]
    return lambd, X


def Shift_Embedding(lambd, X, order, coef, d):
    # lambd, X: top-L eigen-decomposition
    # order: a number indicating the order
    # coef: a vector of length order, indicating the weights for each order
    # d: preset embedding dimension
    # return: content/context embedding vectors
    lambd_H = Eigen_Reweighting(lambd,order,coef)             # High-order transform
    temp_index = np.absolute(lambd_H).argsort()[::-1]         # select top-d
    temp_index = temp_index[:d+1]
    lambd_H = lambd_H[temp_index]
    lambd_H_temp = np.sqrt(np.absolute(lambd_H))
    U = np.dot(X[:,temp_index], np.diag(lambd_H_temp))        # Calculate embedding
    V = np.dot(X[:,temp_index], np.diag(np.multiply(lambd_H_temp, np.sign(lambd_H))))
    return U, V

def AROPE(A, d, order, weights):
    # A: adjacency matrix A or its variations, sparse scipy matrix
    # d: dimensionality
    # r different high-order proximity:
    # order: 1 x r vector, order of the proximity
    # weights: 1 x r list, each containing the weights for one high-order proximity
    # return: 1 x r list, each containing the embedding vectors
    A = A.asfptype()
    lambd, X = Eigen_TopL(A, d)
    r = len(order)
    U_output, V_output = [], []
    for i in range(r):
        U_temp, V_temp = Shift_Embedding(lambd, X, order[i], weights[i], d)
        U_output.append(U_temp)
        V_output.append(V_temp)
    return U_output, V_output
