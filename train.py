import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import torch
import pandas as pd
from scipy import io
from datasets import Meta_data, load_xuzhou_mat
from training import fit
from GLSL_Net import GLSL, Network
from losses import MDTripletLoss
from torch.optim import lr_scheduler
import torch.optim as optim
import random
import time

def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
start = time.time()

def main():
    cuda = torch.cuda.is_available()
    setup_seed(0)
    # Load Salinas dataset
    data_path = './data/xuzhou/xuzhou.mat'#/Salinas/Salinas.mat,/xuzhou/xuzhou.mat,/Indian/Indian_pines.mat
    gt_path = './data/xuzhou/xuzhou_gt.mat'#/Salinas/Salinas_gt.mat,/xuzhou/xuzhou_gt.mat,/Indian/Indian_pines_gt.mat
    train_data, train_gt = load_xuzhou_mat(data_path, gt_path, data_processing=True)

    # Preprocess ground truth
    train_gt0 = io.loadmat(gt_path)['xuzhou_gt'].astype(np.float32).flatten() #salinas_gt,xuzhou_gt,Indian_pines_gt
    train_gt0 = pd.DataFrame(train_gt0).replace(0, np.NAN) # 将所有标签值为0的元素替换为NaN
    train_gt0.dropna(inplace=True)  # 保留的行只包含目标类像素的标签
    train_gt0 = train_gt0.values.reshape(1, -1)[0]

    train_gt = train_gt.astype(np.float32).flatten()

    # Normalize the data
    train_data = np.float32(train_data)
    max1 = np.amax(train_data)
    min1 = np.amin(train_data)
    train_data = (train_data - min1) / max1

    train_data = train_data.reshape(-1, train_data.shape[2])#将数据集转换为二维数组
    train_dataset = Meta_data(train_data, train_gt, train_gt0)

    # DataLoader setup
    batch_size = 64
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # Model and training parameters
    margin = 1.
    embedding_net = GLSL(189)
    model = Network(embedding_net)
    if cuda:
        model.cuda()
    loss_fn = MDTripletLoss(margin)
    lr = 1e-3#1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)
    n_epochs = 10#45
    log_interval = 1

    # Train the model
    fit(train_loader, train_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)
    print("Time:", time.time() - start)

if __name__ == "__main__":
    main()