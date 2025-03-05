import numpy as np
from scipy import io
from torch.utils.data import Dataset

class Meta_data(Dataset):
    """
   每个样本（锚点），随机选择一个正样本和一个负样本
    """
    def __init__(self, train_data, train_label, train_zero, train=True):
        self.train = train
        if self.train:
            self.train_labels = train_label
            self.train_data = train_data
            self.train_label0 = train_zero
            self.labels_set = set(self.train_label0)
            self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                     for label in self.labels_set}
            self.label_index = np.nonzero(self.train_labels)
            self.label_index = np.array([x for x in zip(self.label_index)])
            self.label_index = np.squeeze(self.label_index)

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}
            random_state = np.random.RandomState(29)
            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        #任意选择特定标签的样本
        # if self.train:
        #     img1, label1 = self.train_data[index], self.train_labels[index].item()
        #     # 初始化positive_index以避免未定义错误
        #     positive_index = index  # 这里初始化为index，但最终应该被替换
        #
        #     # 确保正样本来自相同类别（15或16）
        #     positive_label_options = [1, 2]
        #     if label1 in positive_label_options:
        #         while True:
        #             positive_index = np.random.choice(self.label_to_indices[label1])
        #             if positive_index != index:  # 确保正样本不是自身
        #                 break
        #
        #     # 负样本应来自另一个类别（如果label1是15，则负样本来自16，反之亦然）
        #     negative_label = 1 if label1 == 2 else 2
        #     negative_index = np.random.choice(self.label_to_indices[negative_label])
        #
        #     img2 = self.train_data[positive_index]
        #     img3 = self.train_data[negative_index]
        if self.train:
            index = self.label_index[index]
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        img3 = np.expand_dims(img3, axis=0)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.train_label0)

def load_xuzhou_mat(data_path, gt_path, data_processing=True):
    data = io.loadmat(data_path)['xuzhou'].astype(np.float32)
    labels = io.loadmat(gt_path)['xuzhou_gt']

    if data_processing:
        # 189bands-xuzhou
        bands = np.concatenate((np.arange(6, 32),
                                np.arange(75, 136),
                                np.arange(197, 206),
                                np.arange(213, 252),
                                np.arange(366, 420)))
        data = data[:, :, bands]

        # 46 bands-xuzhou
        # bands = np.concatenate((np.arange(20, 30),
        #                         np.arange(40, 55),
        #                         np.arange(97, 103),
        #                         np.arange(113, 120),
        #                         np.arange(209, 217)))
        # data = data[:, :, bands]

        # 200bands
        # bands = np.concatenate((np.arange(7, 32),
        #                         np.arange(33, 96),
        #                         np.arange(97, 106),
        #                         np.arange(109, 152),
        #                         np.arange(160, 220)))
        # data = data[:, :, bands]

        #224bands_xuzou
        # bands = np.concatenate((np.arange(7, 32),
        #                     np.arange(33, 96),
        #                     np.arange(97, 106),
        #                     np.arange(109, 152),
        #                     np.arange(260, 344)))
        # data = data[:, :, bands]

    print('data shape:', data.shape)
    print('label shape:', labels.shape)
    return data, labels

