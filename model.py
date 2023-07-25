
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import utils
from sklearn import cluster
import pickle
import scipy.sparse as sparse
import time
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from metrics.cluster.accuracy import clustering_accuracy
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_kernels
from scipy.sparse import csgraph
import argparse
import random
from scipy.sparse.linalg import svds
from tqdm import tqdm
import os
import csv
from munkres import Munkres
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score, accuracy_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# class conv_feature(nn.Module):
#     def __init__(self):
#         super(conv_feature, self).__init__()
#         self.relu = nn.ReLU()
#         # self.linear1 = nn.Linear(in_features=64, out_features=1024)
#         self.linear1 = nn.Linear(in_features=576, out_features=1024)
#         # self.linear1 = nn.Linear(in_features=1600, out_features=1024)  # patch 15*15*15的IP数据
#
#         self.linear2 = nn.Linear(in_features=1024, out_features=1024)
#         self.linear3 = nn.Linear(in_features=1024, out_features=1024)
#
#         self.conv1_3d = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(5, 5, 5))
#         self.conv2_3d = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
#
#         self.conv1_2d = nn.Conv2d(in_channels=288, out_channels=64, kernel_size=(5, 5))
#         # self.conv1_2d = nn.Conv2d(in_channels=768, out_channels=64, kernel_size=(5, 5))  # IP
#         # self.conv2_2d = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3))  # IP
#
#     def forward(self, x):
#
#         x = self.relu(self.conv1_3d(x))
#         # print(x.shape)
#         x = self.relu(self.conv2_3d(x))
#         # print(x.shape)
#         x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
#         x = self.relu(self.conv1_2d(x))
#         # x = self.relu(self.conv2_2d(x))
#         # print(x.shape)
#
#         x = x.reshape(x.shape[0], -1)
#         # print(x.shape)
#
#         x1 = self.relu(self.linear1(x))
#         x2 = self.linear2(x1)
#         x3 = self.relu(x1 + x2)
#         x4 = self.linear3(x3)
#         x = torch.tanh_(x3 + x4)
#
#         return x
class conv_feature(nn.Module):
    def __init__(self, bands, height, width):
        super(conv_feature, self).__init__()
        self.bands = bands
        self.height = height
        self.width = width
        self.relu = nn.ReLU()
        # self.linear1 = nn.Linear(in_features=64, out_features=1024)
        self.linear1 = nn.Linear(in_features=64*(self.height-10)*(self.width-10), out_features=1024)
        # self.linear1 = nn.Linear(in_features=1600, out_features=1024)  # patch 15*15*15的IP数据

        self.linear2 = nn.Linear(in_features=1024, out_features=1024)
        self.linear3 = nn.Linear(in_features=1024, out_features=1024)

        self.conv1_3d = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(5, 5, 5))
        self.conv2_3d = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))

        self.conv1_2d = nn.Conv2d(in_channels=32*(self.bands-6), out_channels=64, kernel_size=(5, 5))
        # self.conv1_2d = nn.Conv2d(in_channels=768, out_channels=64, kernel_size=(5, 5))  # IP
        # self.conv2_2d = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3))  # IP

    def forward(self, x):

        x = self.relu(self.conv1_3d(x))
        # print(x.shape)
        x = self.relu(self.conv2_3d(x))
        # print(x.shape)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.relu(self.conv1_2d(x))
        # x = self.relu(self.conv2_2d(x))
        # print(x.shape)

        x = x.reshape(x.shape[0], -1)
        # print(x.shape)

        x1 = self.relu(self.linear1(x))
        x2 = self.linear2(x1)
        x3 = self.relu(x1 + x2)
        x4 = self.linear3(x3)
        x = torch.tanh_(x3 + x4)

        return x

# class conv_feature_houston(nn.Module):
#     def __init__(self):
#         super(conv_feature_houston, self).__init__()
#         self.relu = nn.ReLU()
#         self.linear1 = nn.Linear(in_features=576, out_features=1024)
#         # self.linear1 = nn.Linear(in_features=576, out_features=1024)
#         # self.linear1 = nn.Linear(in_features=1600, out_features=1024)  # patch 15*15*15的IP数据
#
#         self.linear2 = nn.Linear(in_features=1024, out_features=1024)
#         self.linear3 = nn.Linear(in_features=1024, out_features=1024)
#
#         self.conv1_3d = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3))
#         self.conv2_3d = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
#
#         self.conv1_2d = nn.Conv2d(in_channels=352, out_channels=64, kernel_size=(3, 3))
#         # self.conv1_2d = nn.Conv2d(in_channels=768, out_channels=64, kernel_size=(5, 5))  # IP
#         # self.conv2_2d = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3))  # IP
#
#     def forward(self, x):
#         x = self.relu(self.conv1_3d(x))
#         x = self.relu(self.conv2_3d(x))
#         x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
#         x = self.relu(self.conv1_2d(x))
#         x = x.reshape(x.shape[0], -1)
#         x1 = self.relu(self.linear1(x))
#         x2 = self.linear2(x1)
#         x3 = self.relu(x1 + x2)
#         x4 = self.linear3(x3)
#         x = torch.tanh_(x3 + x4)
#
#         return x

class conv_feature_houston(nn.Module):
    def __init__(self, bands, height, width):
        super(conv_feature_houston, self).__init__()
        self.bands = bands
        self.height = height
        self.width = width
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features=64*(self.width-6)*(self.height-6), out_features=1024)
        # self.linear1 = nn.Linear(in_features=576, out_features=1024)
        # self.linear1 = nn.Linear(in_features=1600, out_features=1024)  # patch 15*15*15的IP数据

        self.linear2 = nn.Linear(in_features=1024, out_features=1024)
        self.linear3 = nn.Linear(in_features=1024, out_features=1024)

        self.conv1_3d = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=(3, 3, 3))
        self.conv2_3d = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))

        self.conv1_2d = nn.Conv2d(in_channels=32*(self.bands-4), out_channels=64, kernel_size=(3, 3))
        # self.conv1_2d = nn.Conv2d(in_channels=768, out_channels=64, kernel_size=(5, 5))  # IP
        # self.conv2_2d = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(3, 3))  # IP

    def forward(self, x):
        x = self.relu(self.conv1_3d(x))
        x = self.relu(self.conv2_3d(x))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.relu(self.conv1_2d(x))
        x = x.reshape(x.shape[0], -1)
        x1 = self.relu(self.linear1(x))
        x2 = self.linear2(x1)
        x3 = self.relu(x1 + x2)
        x4 = self.linear3(x3)
        x = torch.tanh_(x3 + x4)

        return x


class AdaptiveSoftThreshold(nn.Module):
    def __init__(self, dim):
        super(AdaptiveSoftThreshold, self).__init__()
        self.dim = dim
        self.register_parameter("bias", nn.Parameter(torch.from_numpy(np.zeros(shape=[self.dim])).float()))

    def forward(self, c):
        return torch.sign(c) * torch.relu(torch.abs(c) - self.bias)


class GC_ResSENet(nn.Module):

    def __init__(self, imgname, data):
        super(GC_ResSENet, self).__init__()

        self.shrink = 1.0 / 1024
        batch_size, _, bands, height, width = data.shape
        if imgname == "Houston":
            self.conv = conv_feature_houston(bands, height, width)
        else:
            self.conv = conv_feature(bands, height, width)
        self.thres = AdaptiveSoftThreshold(1)

    def features_extract(self, X):
        x = self.conv(X)
        # print(x.shape)
        return x

    def get_coeff(self, xi, xj):
        c = self.thres(xi.mm(xj.t()))
        return self.shrink * c

    def forward(self, X1, X2):
        u = self.features_extract(X1)
        v = self.features_extract(X2)
        out = self.get_coeff(u, v)
        return out


def regularizer(c, lmbd=1.0):
    return lmbd * torch.abs(c).sum() + (1.0 - lmbd) / 2.0 * torch.pow(c, 2).sum()


def adjacent_mat(x, n_neighbors=11, sparse=False):
    """
    Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
    :param x: array like: n_sample * n_feature
    :return:
    """
    A = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=True).toarray()
    print(np.unique(A))
    # print(A)
    A = A * np.transpose(A)
    # print('A:', A)
    D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
    normlized_A = np.dot(np.dot(D, A), D)
    if sparse:
        normlized_A = torch.from_numpy(normlized_A).to_sparse_csr()
    else:
        normlized_A = torch.from_numpy(normlized_A)
    return normlized_A


def lap_matrix(x):
    # A = kneighbors_graph(X.reshape(X.shape[0], -1), n_neighbors=10, include_self=True, n_jobs=3).toarray()
    # A_ = kneighbors_graph(X.reshape(X.shape[0], -1), n_neighbors=5, include_self=True, n_jobs=8).toarray()
    # A_ = 0.5 * (A_ + A_.T)
    A_ = pairwise_kernels(x.reshape(x.shape[0], -1), metric='rbf', gamma=1., n_jobs=8)
    A = 0.5 * (A_ + A_.T)
    # A_[np.nonzero(A_)] = A[np.nonzero(A_)]
    L = csgraph.laplacian(A, normed=True)
    return L


def get_sparse_rep(gsenet, data, non_zeros=1000):
    N = data.shape[0]
    non_zeros = min(N, non_zeros)
    C = torch.empty([N, N])  # 创建一个[N, N]大小的张量

    val = []
    indicies = []
    with torch.no_grad():
        gsenet.eval()
        chunk = data.cuda()
        u = gsenet.features_extract(chunk)
        temp = gsenet.get_coeff(u, u)
        C = temp.cpu()

        # 数据本身的自表达系数为0
        rows = list(range(N))
        cols = [j for j in rows]
        C[rows, cols] = 0.0

        _, index = torch.topk(torch.abs(C), dim=1, k=non_zeros)

        val.append(C.gather(1, index).reshape([-1]).cpu().data.numpy())  # 取出上步的值，大小为[batch_size*non_zeros]
        index = index.reshape([-1]).cpu().data.numpy()
        indicies.append(index)

    val = np.concatenate(val, axis=0)
    indicies = np.concatenate(indicies, axis=0)
    indptr = [non_zeros * i for i in range(N + 1)]

    C_sparse = sparse.csr_matrix((val, indicies, indptr), shape=[N, N])
    return C_sparse


def evaluate(gsenet, train_data, labels, num_subspaces, non_zeros, folder, im_name, spectral_dim=5):
    # C = get_C(gsenet=gsenet, data=train_data)
    C_sparse = get_sparse_rep(gsenet=gsenet, data=train_data, non_zeros=non_zeros)
    Coef = C_sparse.todense()
    print('C compute done!')
    # Coef = thrC(Coef, 0.25)
    y_pre, C = post_proC(Coef, num_subspaces, 8, 18, im_name, spectral_dim)
    with open('plot/{}-labels.pkl'.format(im_name), 'wb') as f:
        pickle.dump(y_pre, f)
    print('spectral_clustering done!')
    # acc = clustering_accuracy(labels, y_pre)
    # nmi = normalized_mutual_info_score(labels, y_pre, average_method='geometric')
    # ari = adjusted_rand_score(labels, y_pre)
    acc, nmi, kappa, _ = cluster_accuracy(labels, y_pre)
    return acc, nmi, kappa


def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp


def cluster_accuracy(y_true, y_pre):
    y_best = best_match(y_true, y_pre)
    # # calculate accuracy
    err_x = np.sum(y_true[:] != y_best[:])
    missrate = err_x.astype(float) / (y_true.shape[0])
    acc = 1. - missrate
    nmi = normalized_mutual_info_score(y_true, y_pre)
    kappa = cohen_kappa_score(y_true, y_best)
    ca = class_acc(y_true, y_best)
    return acc, nmi, kappa, ca


def best_match(y_true, y_pre):
    Label1 = np.unique(y_true)
    nClass1 = len(Label1)
    Label2 = np.unique(y_pre)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = y_true == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = y_pre == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    y_best = np.zeros(y_pre.shape)
    for i in range(nClass2):
        y_best[y_pre == Label2[i]] = Label1[c[i]]
    return y_best


def class_acc(y_true, y_pre):
    """
    calculate each class's acc
    :param y_true:
    :param y_pre:
    :return:
    """
    ca = []
    for c in np.unique(y_true):
        y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
        y_c_p = y_pre[np.nonzero(y_true == c)]
        acurracy = accuracy_score(y_c, y_c_p)
        ca.append(acurracy)
    ca = np.array(ca)
    return ca


def post_proC(C, K, d, alpha, im_name, spectral_dim):
    if im_name == 'Indian_pines_corrected':
        C_knn = kneighbors_graph(np.array(C), 6, mode='connectivity', include_self=False, n_jobs=10)
        L = 0.5 * (C_knn + C_knn.T)
        grp = utils.spectral_clustering(L, K, spectral_dim)
    else:
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        grp = utils.spectral_clustering(L, K, K)
        # spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
        #                                       assign_labels='discretize')
        # spectral.fit(L)
        # grp = spectral.fit_predict(L)
    return grp, L


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

