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
from thop import profile
from torchsummary import summary


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
    patchesLabels = np.zeros((NoZeroNumber))
    patchIndex = 0
    spatial_list = np.zeros((NoZeroNumber, 2))
    # 遍历数据，得到（25,25,30）的数据和数据标签
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            if y[r - margin, c - margin] == 0:
                continue
            else:
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = y[r - margin, c - margin]
                spatial_list[patchIndex, :] = [r, c]
                patchIndex = patchIndex + 1
    patchesLabels -= 1
    # 去除标签为0的无效数据,且标签减1，数据标签从0开始
    print("---遍历完成---")
    # if removeZeroLabels:
    #     patchesData = patchesData[patchesLabels > 0, :, :, :]
    #     patchesLabels = patchesLabels[patchesLabels > 0]
    #     patchesLabels -= 1
    return patchesData, patchesLabels, spatial_list


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
    root = 'datasets/'
    DATA = ['SalinasA_corrected', 'Indian_pines_corrected', 'PaviaU', 'Houston']
    GT = ['SalinasA_gt', 'Indian_pines_gt', 'PaviaU_gt', 'Houston_gt']
    # DATA = ['SalinasA_corrected']
    # GT = ['SalinasA_gt']
    for im_, gt_ in zip(DATA, GT):

        img_path = root + im_ + '.mat'
        gt_path = root + gt_ + '.mat'
        print(img_path)

        # NEIGHBORING_SIZE = 13
        # nb_comps = 15
        img, gt = load_data(img_path, gt_path)

        # 创建结果文件夹
        folder = "{}_result".format(im_)
        if not os.path.exists(folder):
            os.mkdir(folder)

        if im_ == 'PaviaU':
            img, gt = img[150:350, 100:200, :], gt[150:350, 100:200]
            # my para
            K = 60  # 计算A值用到的
            learning_rate = 1e-3
            epochs = 5000
            lmbd = 0.9
            gamma_r = 50
            gamma_g = 200
            nb_comps = 21
            NEIGHBORING_SIZE = 11

        if im_ == 'Indian_pines_corrected':
            img, gt = img[30:115, 24:94, :], gt[30:115, 24:94]
            K = 30
            learning_rate = 3e-3
            epochs = 5000
            lmbd = 0.9
            gamma_r = 200
            gamma_g = 200
            nb_comps = 21
            NEIGHBORING_SIZE = 15

        if im_ == 'SalinasA_corrected':
            LMBD = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            K = 60
            learning_rate = 1e-3
            epochs = 5000
            lmbd = 0.9
            gamma_r = 200
            gamma_g = 200
            nb_comps = 15
            NEIGHBORING_SIZE = 13

        if im_ == 'Houston':
            LMBD = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            print(img.shape, gt.shape)
            img, gt = img[:, 0:680, :], gt[:, 0:680]
            K = 30
            learning_rate = 2e-3
            epochs = 5000
            lmbd = 0.9
            gamma_r = 200
            gamma_g = 200
            nb_comps = 15
            NEIGHBORING_SIZE = 9

        n_row, n_column, n_band = img.shape
        img_scaled = minmax_scale(img.reshape(n_row * n_column, n_band)).reshape(img.shape)
        n_row, n_column, n_band = img_scaled.shape
        print('nb_comps, size', nb_comps, NEIGHBORING_SIZE)
        # perform PCA
        pca = PCA(n_components=nb_comps)
        img = pca.fit_transform(img_scaled.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, nb_comps))
        print('pca shape: %s, percentage: %s' % (img.shape, np.sum(pca.explained_variance_ratio_)))
        x_patches, y_, spatial_list = createImageCubes(img, gt, NEIGHBORING_SIZE)
        # print('spatial_list:', spatial_list.shape)
        print(np.unique(y_))
        train_data = torch.from_numpy(x_patches).float()
        train_data = train_data.permute(0, 3, 1, 2)  # x_patch=(n_samples, n_band, n_width, n_height)
        train_data = train_data.unsqueeze(dim=1)  # 三维卷积
        # x_patches = minmax_scale(x_patches.reshape(x_patches.shape[0], -1)).reshape(x_patches.shape)
        train_data = utils.p_normalize(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        # print('img shape:', img.shape)
        # print('img_patches_nonzero:', x_patches.shape)
        # print('train data shape:', train_data.shape)
        non_zeros = train_data.shape[0]
        # print('non_zeros:', non_zeros)
        # n_samples, n_width, n_height, n_band = x_patches.shape

        y = standardize_label(y_)
        # print('x_patches shape: %s, labels: %s' % (x_patches.shape, np.unique(y)))
        A0 = model.adjacent_mat(train_data.reshape(train_data.shape[0], -1).numpy(), K)
        A0 = A0.float().cuda()

        N_CLASSES = np.unique(y).shape[0]  # wuhan : 5  Pavia : 6  Indian : 8  KSC : 10  SalinasA : 6 PaviaU : 8

        model.same_seeds(0)
        # 网络及其参数定义
        # gsenet = model.GSENet().cuda()
        GC_ResSENet = model.GC_ResSENet(im_, train_data).cuda()  # 专门对IP数据的网络，1个三维卷积，两个二维卷积
        # train_data = train_data.cuda()
        # flops, params = profile(GC_ResSENet, inputs=(train_data,train_data))
        # print("FLOPs:", flops)
        # print(train_data.shape)
        # # 计算模型参数数量
        # num_params = sum(p.numel() for p in GC_ResSENet.parameters() if p.requires_grad)
        # print("模型参数数量为：", num_params)
        # # 计算模型大小（以MB为单位）
        # model_size = num_params * 4 / (1024 * 1024)
        # print("模型大小为：%.6f MB" % model_size)

        optimizer = optim.Adam(GC_ResSENet.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        time_start = time.time()
        pbar = tqdm(range(epochs), ncols=120)
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}")
            randidx = torch.randperm(train_data.shape[0])  # 生成一个[0,train_data.shape[0]-1]的随机序列  打扰样本顺序
            A = A0[randidx]
            A = A[:, randidx]

            GC_ResSENet.train()
            batch = train_data[randidx].cuda()
            g_batch = A.mm(batch.reshape(batch.shape[0], -1))

            u_data = GC_ResSENet.features_extract(batch)

            c = GC_ResSENet.get_coeff(u_data, u_data)  # alpha*T(uV)
            rec_batch = c.mm(batch.reshape(batch.shape[0], -1))
            g_batch = c.mm(g_batch)  # CAX
            reg = model.regularizer(c, lmbd)
            #
            diag_c = GC_ResSENet.thres((u_data * u_data).sum(dim=1, keepdim=True)) * GC_ResSENet.shrink  # 求cii，自身的自表达系数
            rec_batch = rec_batch - diag_c * batch.reshape(batch.shape[0], -1)  # 减去自身的表达系数，不能用自己本身来代表
            g_batch = g_batch - diag_c * g_batch
            reg = reg - model.regularizer(diag_c, lmbd)  # 减去已经正则化的自表达系数

            rec_loss = torch.sum(torch.pow(batch.reshape(batch.shape[0], -1) - rec_batch, 2))  # 自表达系数的重建损失
            Grec_loss = torch.sum(torch.pow(batch.reshape(batch.shape[0], -1) - g_batch, 2))

            loss = (0.5 * gamma_r * rec_loss + reg + 0.5 * gamma_g * Grec_loss) / train_data.shape[0]  # 总损失

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(GC_ResSENet.parameters(), 0.001)
            optimizer.step()

            scheduler.step()

            # print('epoch:{}\tloss:{}\trec_loss:{}\treg_loss:{}\tGrec_loss:{}'.format(epoch, loss.item(), rec_loss.item(
            # ) / train_data.shape[0], reg.item() / train_data.shape[0], Grec_loss.item() / train_data.shape[0]))
            pbar.set_postfix(loss="{:3.4f}".format(loss.item()),
                             rec_loss="{:3.4f}".format(rec_loss.item() / train_data.shape[0]),
                             reg="{:3.4f}".format(reg.item() / train_data.shape[0]),
                             Grec_loss="{:3.4f}".format(Grec_loss.item() / train_data.shape[0]))
        print("Evaluating on train data...")
        acc, nmi, kappa = model.evaluate(GC_ResSENet, train_data=train_data.cuda(), labels=y, num_subspaces=N_CLASSES,
                                       non_zeros=non_zeros, folder=folder, im_name=im_)
        print("ACC-{:.6f}, NMI-{:.6f}, Kappa-{:.6f}".format(acc, nmi, kappa))
        print('{}-times: '.format(im_), time.time()-time_start)
        # with open('{}/GCressenet-ACC{:.6f}-lmbd:{}.pth'.format(folder, acc, lmbd), 'wb') as f:
        #     torch.save(GC_ResSENet.state_dict(), f)
    print('done!')