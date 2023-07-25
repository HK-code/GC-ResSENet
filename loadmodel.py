import argparse
import pickle

import scipy.io
import torch
import torch.nn as nn
import model
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import time
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import utils
import numpy as np
import torch.optim as optim
from collections import Counter

from yaml_config_hook import yaml_config_hook


def load_data(img_path, gt_path):
    if img_path[-3:] == 'mat':
        import scipy.io as sio
        img_mat = sio.loadmat(img_path)
        gt_mat = sio.loadmat(gt_path)
        img_keys = img_mat.keys()
        gt_keys = gt_mat.keys()
        img_key = [k for k in img_keys if k != '__version__' and k != '__header__' and k != '__globals__']
        gt_key = [k for k in gt_keys if k != '__version__' and k != '__header__' and k != '__globals__']
        return img_mat.get(img_key[0]).astype('float64'), gt_mat.get(gt_key[0]).astype('int8')


def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X  # 给数据加上一圈的0值
    return newX


def calNoZeroNumber(y):
    count = 0
    for r in range(y.shape[0]):
        for c in range(y.shape[1]):
            if y[r, c] == 0:
                continue
            else:
                count += 1
    return count


def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 统计非零标签像素的个数
    NoZeroNumber = calNoZeroNumber(y)
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    # 以该数据为例创建一个（512*217,25,25,30）的四维矩阵用于存储每一个（25,25,30）的数据块
    # 并创建一个（512*127）的标签
    patchesData = np.zeros((NoZeroNumber, windowSize, windowSize, X.shape[2]))
    print("patchesData shape is", patchesData.shape)
    patchesLabels = np.zeros(NoZeroNumber)
    patchIndex = 0
    # 遍历数据，得到（25,25,30）的数据和数据标签
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if y[r - margin, c - margin] == 0:
                continue
            else:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                patchIndex = patchIndex + 1
    patchesLabels -= 1
    # 去除标签为0的无效数据,且标签减1，数据标签从0开始
    print("---遍历完成---")
    # if removeZeroLabels:
    #     patchesData = patchesData[patchesLabels > 0, :, :, :]
    #     patchesLabels = patchesLabels[patchesLabels > 0]
    #     patchesLabels -= 1
    return patchesData, patchesLabels


def standardize_label(y):
    """
    standardize the classes label into 0-k
    :param y:
    :return:
    """
    import copy
    classes = np.unique(y)
    standardize_y = copy.deepcopy(y)
    for i in range(classes.shape[0]):
        standardize_y[np.nonzero(y == classes[i])] = i
    return standardize_y


def order_by_diag(labels):
    # print(labels.shape)
    # print(np.unique(labels).shape)
    order = np.array([])
    for i in np.unique(labels):
        order = np.append(order, np.nonzero(labels == i)[0])
    #     print(np.nonzero(labels == i)[0].shape[0])  # 对照unique数组，依次统计每个元素出现的次数
    # print(order.shape)
    return order.astype(dtype=np.int16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config/PaviaU_config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    root, im_, gt_ = args.root, args.im_, args.gt_

    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'
    print(img_path)
    img, gt = load_data(img_path, gt_path)

    # 创建结果文件夹
    folder = "{}_result".format(im_)
    if not os.path.exists(folder):
        os.mkdir(folder)

    if im_ == 'PaviaU':
        img, gt = img[150:350, 100:200, :], gt[150:350, 100:200]
        model_path = args.model_path
        NEIGHBORING_SIZE = args.NEIGHBORING_SIZE
        nb_comps = args.nb_comps

    if im_ == 'Indian_pines_corrected':
        img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]
        model_path = args.model_path
        NEIGHBORING_SIZE = args.NEIGHBORING_SIZE
        nb_comps = args.nb_comps

    if im_ == 'SalinasA_corrected':
        model_path = args.model_path
        NEIGHBORING_SIZE = args.NEIGHBORING_SIZE
        nb_comps = args.nb_comps

    if im_ == 'Houston':
        print(img.shape, gt.shape)
        img, gt = img[:, 0:680, :], gt[:, 0:680]
        model_path = args.model_path
        NEIGHBORING_SIZE = args.NEIGHBORING_SIZE
        nb_comps = args.nb_comps

    n_row, n_column, n_band = img.shape
    img_scaled = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape(img.shape)

    # perform PCA
    pca = PCA(n_components=nb_comps)
    img = pca.fit_transform(img_scaled.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, nb_comps))
    print('pca shape: %s, percentage: %s' % (img.shape, np.sum(pca.explained_variance_ratio_)))
    x_patches, y_ = createImageCubes(img, gt, NEIGHBORING_SIZE)
    print(np.unique(y_))
    train_data = torch.from_numpy(x_patches).float()
    train_data = train_data.permute(0, 3, 1, 2)  # x_patch=(n_samples, n_band, n_width, n_height)
    train_data = train_data.unsqueeze(dim=1)  # 三维卷积
    # x_patches = minmax_scale(x_patches.reshape(x_patches.shape[0], -1)).reshape(x_patches.shape)
    train_data = utils.p_normalize(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
    print('img shape:', img.shape)
    print('img_patches_nonzero:', x_patches.shape)
    print('train data shape:', train_data.shape)
    non_zeros = train_data.shape[0]
    print('non_zeros:', non_zeros)
    n_samples, n_width, n_height, n_band = x_patches.shape

    y = standardize_label(y_)
    print('x_patches shape: %s, labels: %s' % (x_patches.shape, np.unique(y)))

    N_CLASSES = np.unique(y).shape[0]  # wuhan : 5  Pavia : 6  Indian : 8  KSC : 10  SalinasA : 6 PaviaU : 8

    model.same_seeds(0)
    # 网络及其参数定义
    GC_ResSENet = model.GC_ResSENet(im_, train_data).cuda()  # 专门对IP数据的网络，1个三维卷积，两个二维卷积
    GC_ResSENet.load_state_dict(torch.load(model_path))  # salinasA
    acc, nmi, kappa = model.evaluate(GC_ResSENet, train_data=train_data.cuda(), labels=y, num_subspaces=N_CLASSES,
                                   non_zeros=non_zeros, folder=folder, im_name=im_, spectral_dim=5)
    print("ACC-{:.6f}, NMI-{:.6f}, kappa-{:.6f}".format(acc, nmi, kappa))
    print('done!')
