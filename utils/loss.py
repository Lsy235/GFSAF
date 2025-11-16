import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LTwoLoss(nn.Module):
    def __init__(self, batch_size):
        super(LTwoLoss, self).__init__()
        self. batch_size = batch_size

    def foward_ZLTwo(self, z):
        return torch.mean(torch.norm(z, dim=1))

class SimLoss(nn.Module):
    def __init__(self, batch_size):
        super(SimLoss, self).__init__()
        self.batch_size = batch_size

    def forward_orthogonal(self, z1, z2):
        cos_sim = F.cosine_similarity(z1, z2, dim=1)
        return cos_sim.mean()

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        # self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        # self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_loss(self, h_i, h_j, batch_size=256):
        self.batch_size = batch_size
        self.batch_size = h_i.shape[0]

        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # print(f"h shape: {h.shape}, h_i shape: {h_i.shape}, h_j shape: {h_j.shape}, sim shape: {sim.shape}, bs: {self.batch_size}, sim_i_j shape:{sim_i_j.shape}, sim_j_i shape:{sim_j_i.shape}")
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
