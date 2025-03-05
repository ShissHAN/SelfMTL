import os
from torch import nn, optim
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
from typing import List
import numpy as np
import torch
import torch.utils.data as data
from scipy import io
from GLSL_Net import GLSL, Network
from utils import cos_sim, HyperX
from utils import plot_roc_curve
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.functional as F
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
setup_seed(3327)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label, size_average=True):
        distance_positive = (output1 - output2).pow(2).sum(1)  # .pow(.5)
        loss1 = F.relu((label) * torch.pow(distance_positive, 2) + (1-label) * torch.pow(torch.clamp(
            self.margin - distance_positive, min=0.0), 2))
        return loss1.mean() if size_average else loss1.sum()

#Sandiego  189bands
data_name = 'Sandiego'
hyperdata_path = './data/Sandiego/sandiego.mat'
dgt_path = './data/Sandiego/groundtruth.mat'
prior_path = './data/Sandiego/tgt_sandiego_2.mat'
hyperdata = io.loadmat(hyperdata_path)['data']
dgt = io.loadmat(dgt_path)['gt']
prior = io.loadmat(prior_path)['tgt']
# ...............................................................................................
hyperdata = np.float32(hyperdata)
max1 = np.amax(hyperdata)
min1 = np.amin(hyperdata)
hyperdata = (hyperdata-min1)/max1
detect_dataset = HyperX(hyperdata, dgt)
detect_loader = data.DataLoader(detect_dataset, batch_size=1)
prior = np.float32(prior)
max2 = np.amax(prior)
min2 = np.amin(prior)
prior = (prior-min2)/max2
#微调 AV-(10, 88),负样本(3, 88)
k_path = (10, 88)
positive_sample = hyperdata[k_path[0], k_path[1]]
positive_sample = np.expand_dims(positive_sample, axis=0)
positive_label = 1
k_path_negative = (3, 88)
negative_sample = hyperdata[k_path_negative[0], k_path_negative[1]]
negative_sample = np.expand_dims(negative_sample, axis=0)
negative_label = 0

class CustomDataset(data.Dataset):
    def __init__(self, samples, labels):
        assert len(samples) == len(labels),'samples and labels must have the same length.'
        self.samples = samples
        self.labels = labels
    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]
        return sample, label
    def __len__(self):
        return len(self.samples)

samples = [positive_sample, negative_sample]
labels = [positive_label, negative_label]
train_dataset = CustomDataset(samples, labels)
train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)

model = Network(GLSL(189)).to(device)
model.load_state_dict(torch.load('./data/xu_San_model.ckpt'))
prior = np.expand_dims(prior, axis=0)
prior = torch.from_numpy(prior)
criterion = ContrastiveLoss(margin=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# 训练模型
num_epochs = 13#10
model.train()
prior = prior.to(device)
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = torch.squeeze(images)
        images = torch.unsqueeze(images, 0)
        images = torch.unsqueeze(images, 0)
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output2 = model.get_embedding(images)
        output1 = model.get_embedding(prior)
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")
torch.save(model.state_dict(), './data/Sandiego2_model.pth')
model.load_state_dict(torch.load('./data/Sandiego2_model.pth'))
start = time.time()

#Test
model.eval()
target_detector: List[None] = []
sandiego_feature: List[None] = []
with torch.no_grad():
    prior = prior.to(device)
    prior_output = model.get_embedding(prior)
    prior_output = prior_output.cpu().numpy()
    for images, _ in detect_loader:
        images = torch.squeeze(images)
        images = torch.unsqueeze(images, 0)
        images = torch.unsqueeze(images, 0)
        images = images.to(device)
        outputs = model.get_embedding(images)
        outputs = outputs.cpu().numpy()
        detection = cos_sim(prior_output, outputs)
        target_detector.append(detection)
        sandiego_feature.append(outputs)
print("time:",time.time()-start)
target_detector = np.array(target_detector)
target_detector = target_detector.squeeze()
sandiego_feature = np.array(sandiego_feature)
sandiego_feature = sandiego_feature.squeeze()
tdgt = dgt.astype(np.float32)
tdgt = tdgt.reshape(-1)
max3 = np.amax(target_detector)
min3 = np.amin(target_detector)
target_detector = (target_detector - min3)/(max3 - min3)
plot_roc_curve(tdgt, target_detector, data_name)
H, W = dgt.shape
target_detector = np.reshape(target_detector, (H, W))
plt.figure(2)
plt.imshow(target_detector, cmap='gray')
plt.axis('off')
# 保存结果
pathfigure = './result/' + data_name + '.png'
plt.savefig(pathfigure, bbox_inches = 'tight', pad_inches = 0, dpi = 1000)
plt.show()
path = './result/' + data_name + '.mat'
io.savemat(path, {'detect': target_detector})






