import torch
import torch.nn as nn
import torch.nn.functional as F

class MDTripletLoss(nn.Module):
    def __init__(self, margin):
        super(MDTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        distance_pn = (positive - negative).pow(2).sum(1)  # .pow(.5)
        # 选取distance_negative和distance_pn中的较大值
        max_distance_neg_pn = torch.max(distance_negative, distance_pn)
        losses1 = F.relu(distance_positive - max_distance_neg_pn + self.margin)

        return losses1.mean() if size_average else losses1.sum()


class OnlineTripletLoss(nn.Module):
    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        if embeddings.is_cuda:
            triplets = triplets.cuda()
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)
