import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import torch
import torch.nn.functional as F
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.io as io

def compute_neighborhood_sum(mask, i, j, size):
    """
    计算给定位置 (i, j) 的邻域内的标签总和。
    mask: 目标像素的标签矩阵
    i, j: 当前像素位置
    size: 邻域的范围（半径）
    """
    i_start = max(i - size, 0)
    i_end = min(i + size + 1, mask.shape[0])
    j_start = max(j - size, 0)
    j_end = min(j + size + 1, mask.shape[1])
    return mask[i_start:i_end, j_start:j_end].sum().item()

def upsample_and_detect(target_detector, hyperdata_path, d, r):
    target_detector = torch.tensor(target_detector, dtype=torch.float32)
    hyperdata = io.loadmat(hyperdata_path)['data']
    hyperdata = hyperdata.astype(float)
    H, W, C = hyperdata.shape
    # 根据阈值r筛选出目标像素
    target_mask = target_detector > r

    # 将先验光谱d转换为tensor
    #d = torch.tensor(d, dtype=torch.float32)

    # 定义腐蚀和膨胀的卷积核
    erosion_kernel = torch.ones((3, 3), dtype=torch.float32)
    dilation_kernel = torch.ones((7, 7), dtype=torch.float32)
    # 初始化最终结果
    final_target_mask = target_detector.clone()
    for i in range(H):
        for j in range(W):
            if target_mask[i, j]:
                # 计算3x3邻域总和
                neighborhood_sum_3x3 = compute_neighborhood_sum(target_mask, i, j, size=1)
                weight = np.exp(-0.1 * neighborhood_sum_3x3)
                if neighborhood_sum_3x3 < 5:
                    final_target_mask[i, j] = max(final_target_mask[i, j] - weight, 0)  # 降低并确保不低于0.0
                # 腐蚀操作
                eroded_mask = F.conv2d(final_target_mask.unsqueeze(0).unsqueeze(0), erosion_kernel.unsqueeze(0).unsqueeze(0),padding=1)
                eroded_mask = torch.clamp(eroded_mask.squeeze(0).squeeze(0), 0, 1)

                # 膨胀操作
                dilated_mask = F.conv2d(final_target_mask.unsqueeze(0).unsqueeze(0),dilation_kernel.unsqueeze(0).unsqueeze(0), padding=3)
                dilated_mask = torch.clamp(dilated_mask.squeeze(0).squeeze(0), 0, 1)

                # 更新 final_target_mask
                final_target_mask = final_target_mask * dilated_mask - (1 - dilated_mask) * eroded_mask

                # 计算5x5邻域总和
                neighborhood_sum_5x5 = compute_neighborhood_sum(target_mask, i, j, size=3)
                weight = (1 - np.exp(-0.1 * neighborhood_sum_5x5)) * 0.5
                #print(weight)
                if neighborhood_sum_5x5 > 30:
                    i_start = max(i - 2, 0)
                    i_end = min(i + 3, target_mask.shape[0])
                    j_start = max(j - 2, 0)
                    j_end = min(j + 3, target_mask.shape[1])
                    for ni in range(i_start, i_end):
                        for nj in range(j_start, j_end):
                            if target_mask[ni, nj]:
                                final_target_mask[ni, nj] = max(final_target_mask[ni, nj] - weight, 0)  # 降低并确保不低于0.0

    binary_mask = (final_target_mask > 0.9).float()

    # 定义腐蚀和膨胀的卷积核
    erosion_kernel = torch.ones((3, 3), dtype=torch.float32)
    dilation_kernel = torch.ones((7, 7), dtype=torch.float32)

    # 腐蚀操作
    eroded_mask = F.conv2d(binary_mask.unsqueeze(0).unsqueeze(0), erosion_kernel.unsqueeze(0).unsqueeze(0), padding=1)
    eroded_mask = torch.clamp(eroded_mask.squeeze(0).squeeze(0), 0, 1)

    # 膨胀操作
    dilated_mask = F.conv2d(binary_mask.unsqueeze(0).unsqueeze(0), dilation_kernel.unsqueeze(0).unsqueeze(0), padding=3)
    dilated_mask = torch.clamp(dilated_mask.squeeze(0).squeeze(0), 0, 1)

    # 更新 final_target_mask
    final_target_mask = final_target_mask * dilated_mask - (1 - dilated_mask) * eroded_mask

    return final_target_mask.numpy()

data_name = 'Sandiego'
hyperdata_path = './data/Sandiego/sandiego.mat'
prior_path = './data/Sandiego/tgt_sandiego_2.mat'
hyperdata = io.loadmat(hyperdata_path)['data']
d = io.loadmat(prior_path)['tgt']

# 从文件中读取数据
mat_data = scipy.io.loadmat('./result/Sandiego.mat')
target_detector = mat_data['detect']
# 进行归一化处理
min_val = np.min(target_detector)
max_val = np.max(target_detector)
target_detector = (target_detector - min_val) / (max_val - min_val)
plt.imshow(target_detector, cmap='gray')
plt.axis('off')
plt.show()

# 示例阈值
r = 0.6  # 阈值r
final_target_mask = upsample_and_detect(target_detector, hyperdata_path, d, r)

# 进行归一化处理
max_val = np.amax(final_target_mask)
min_val = np.amin(final_target_mask)
final_target_mask = (final_target_mask - min_val) / (max_val - min_val)

# 加载真实标签数据
dgt_path = './data/Sandiego/groundtruth.mat'
dgt = io.loadmat(dgt_path)['gt']
dgt = torch.tensor(dgt, dtype=torch.float32).numpy()
tdgt = dgt.astype(np.float32).reshape(-1)

# 计算AUC
fpr, tpr, threshold = roc_curve(tdgt, final_target_mask.reshape(-1))
fpr = fpr[1:]
tpr = tpr[1:]
threshold = threshold[1:]
roc_auc = auc(fpr, tpr)
auc_f = auc(threshold,fpr)
auc_d = auc(threshold,tpr)
print('roc_auc:', roc_auc)
print('auc_f:', auc_f)
print('auc_d:', auc_d)
print('auc_td:', auc_d + roc_auc)
print('auc_bs:', roc_auc - auc_f)
print('auc_odp:', 1 + auc_d - auc_f)

H, W = dgt.shape
final_target_mask = np.reshape(final_target_mask, (H, W))
plt.figure(2)
plt.imshow(final_target_mask, cmap='gray')
plt.axis('off')

# 保存结果到.mat文件
path = './result/' + 'Sandiego_enhanced' + '.mat'
io.savemat(path, {'detect': final_target_mask})
