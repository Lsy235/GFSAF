import torch
import numpy as np
from .granular import GranularBall, GBList, MVGBList
from .tools import relation_of_views_gblists, merge_tensors, relation_of_views_gblists_tensor, count_crossview_pairs


class GranularContrastiveLoss(torch.nn.Module):
    # 对比学习：让近邻球靠近，非近邻球远离
    # 近邻矩阵相当于标识出正样本和负样本
    def __init__(self, temperature=1.):
        super(GranularContrastiveLoss, self).__init__()
        self.t = temperature

    def forward(self, gblist):
        pos_mask = gblist.affinity()
        neg_mask = 1 - pos_mask
        num_ins = len(gblist)
        idx = torch.arange(0, num_ins)
        # 修正正样本对掩码
        pos_mask[idx, idx] = 0
        x = gblist.get_centers()
        # 计算相似度，这里就是矩阵相乘
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
        # 考虑用cross entropy 重写
        sim_pos = pos_mask * sim_x / self.t
        sim_neg = neg_mask * sim_x / self.t
        exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((num_ins, num_ins))
        expsum_sim = torch.exp(sim_pos) + exp_sim_neg
        # expsum_sim = exp_sim_neg
        loss = -(sim_pos - torch.log(expsum_sim) * pos_mask)

        avg_sim_pos = torch.sum(sim_pos) / torch.sum(pos_mask)
        avg_sim_neg = torch.sum(sim_neg) / (torch.sum(neg_mask))
        return torch.sum(torch.as_tensor(loss)) / num_ins, avg_sim_pos, avg_sim_neg

from .loss import Loss
class MultiviewGCLoss(torch.nn.Module):
    def __init__(self, args, temperature=1.):
        super(MultiviewGCLoss, self).__init__()
        self.t = temperature
        self.args = args
        self.criterion = Loss(args.batch_size, args.class_num, args.temperature, args.device).to(args.device)

    def critenContrativeTolLoss(self, view, num_views, k, batch_size):
        x = view.get_centers()
        cluster_labels = view.y_parts
        device = view.data.device
        # print(f"f: {k}")
        pos_mask = torch.eye(k, device=device)
        for i in range(k):
            for j in range(i+1, k):
                rate_ij, rate_ji = count_crossview_pairs(cluster_labels,num_views=num_views, batch_size=batch_size, i=i, j=j)
                if(rate_ij>0.7 or rate_ji > 0.7):
                    pos_mask[i,j] = 1.
                    pos_mask[j,i] = 1.

        neg_mask = torch.ones_like(pos_mask).to(device) - pos_mask
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
        # 考虑用cross entropy 重写
        # print(f"x shape: {x.shape}, pos_mask: {pos_mask.shape}, sim_x shape: {sim_x.shape}")
        sim_pos = pos_mask * sim_x / self.t
        sim_neg = neg_mask * sim_x / self.t
        exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((k, k))
        expsum_sim = torch.exp(sim_pos) + exp_sim_neg
        # expsum_sim = exp_sim_neg
        return torch.sum(-(sim_pos - torch.log(expsum_sim) * pos_mask)) / pos_mask.sum()

    def critenContrativeLoss(self, view1, view2, mask_i_intra, device):
        # 计算掩码
        # mask_j_intra = views[j].affinity()
        mask_j_intra = torch.eye(len(view2), device=device)
        mask_inter = relation_of_views_gblists_tensor(view1, view2)
        # 两个视图的粒球数量
        ni, nj = len(view1), len(view2)
        # 合并视图内和视图间的掩码矩阵
        pos_mask = merge_tensors(ni, nj, mask_i_intra, mask_inter, mask_inter.T, mask_j_intra).to(device)
        # neg_mask = 1 - pos_mask
        neg_mask = torch.ones_like(pos_mask).to(device) - pos_mask
        num_ins = ni + nj
        # idx = torch.arange(0, num_ins)
        # 修正正样本对掩码
        # pos_mask[idx, idx] = 0
        centers_i = view1.get_centers()
        centers_j = view2.get_centers()
        x = torch.concat((centers_i, centers_j), dim=0)
        # 计算相似度，这里就是矩阵相乘
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
        # 考虑用cross entropy 重写
        sim_pos = pos_mask * sim_x / self.t
        sim_neg = neg_mask * sim_x / self.t
        exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((num_ins, num_ins))
        expsum_sim = torch.exp(sim_pos) + exp_sim_neg
        # expsum_sim = exp_sim_neg
        return torch.sum(-(sim_pos - torch.log(expsum_sim) * pos_mask)) / pos_mask.sum()
    def forward(self, views, num_views, k, batch_size, mode=0):
        if(mode == 1):
            return self.critenContrativeTolLoss(views, num_views, k, batch_size)
        elif(mode == 0):
            # 统一设备
            device = views[0].data.device
            loss = torch.tensor(0., device=device)
            # 两两视图之间进行对比
            num_views = len(views)
            for i in range(num_views):
                # mask_i_intra = views[i].affinity()
                mask_i_intra = torch.eye(len(views[i]), device=device)
                for j in range(i + 1, num_views):
                    view1 = views[i]
                    view2 = views[j]
                    centers_i = view1.get_centers()
                    centers_j = view2.get_centers()
                    xnum = centers_i.shape[0]
                    if(self.args.contrastive_loss == 'InfoNCE'):
                        loss += self.criterion.forward_loss(centers_i, centers_j, batch_size=xnum)
                    else:
                        loss += self.critenContrativeLoss(view1, view2, mask_i_intra, device)
            return loss / (num_views * (num_views - 1) / 2)
