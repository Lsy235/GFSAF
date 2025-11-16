import torch
from torch.nn.functional import one_hot
import numpy as np
from .granular import GranularBall, GBList, contain_same_sample


# 两个视图的粒球集之间的关系
def relation_of_views_gblists(view0: GBList, view1: GBList, t=0.1):
    # 不同视图之间的实例具有对应关系，因此需要建立跨视图粒球之间的联系
    # 一种简单的思路是：直接按照实例对应关系建立联系，为此，需要知道每个粒球中包含哪些样本（索引）
    # 这个关系可以作为对比学习过程的掩码
    n0, n1 = len(view0), len(view1)
    mask = np.zeros((n0, n1), dtype=np.float32)
    for i in range(n0):
        set0 = set(view0[i].indices)
        for j in range(n1):
            # if contain_same_sample(view0[i], view1[j]):
            #     mask[i, j] = 1
            set1 = set(view1[j].indices)
            sub_set = set0 & set1
            if len(sub_set) / len(set0) > t or len(sub_set) / len(set1) > t:
                mask[i, j] = 1
    return torch.from_numpy(mask).to(view0.data.device)

def relation_of_views_gblists_tensor(view0: GBList, view1: GBList, t=0.1):
    y_parts0 = view0.y_parts
    y_parts1 = view1.y_parts
    num_gb = len(view0)
    # n * k
    one_hot0 = one_hot(y_parts0, num_classes=num_gb).float()
    # n * k
    one_hot1 = one_hot(y_parts1, num_classes=num_gb).float()
    mask = one_hot0.T @ one_hot1
    num_gb_set0 = one_hot0.sum(dim=0).view((-1, 1))
    num_gb_set1 = one_hot1.sum(dim=0).view((1, -1))
    num_gb_min = torch.min(num_gb_set0, num_gb_set1)
    mask =  (mask / num_gb_min) > t
    return mask.float()

def count_crossview_pairs(cluster_labels_ori, num_views=3, batch_size=256, i=0, j=1):
    """
    统计属于 cluster i 的样本，其其他视图出现在 cluster j 中的样本数量（反之亦然）
    """
    # print(f"cluster_labels_ori shape: {cluster_labels_ori.shape}")
    cluster_labels = np.array(cluster_labels_ori)
    total = num_views * batch_size  # 应该为 768
    assert cluster_labels.shape[0] == total

    # 构建映射：sample_id -> [view0_label, view1_label, view2_label]
    sample_view_clusters = cluster_labels.reshape(num_views, batch_size).T  # shape: (256, 3)

    # 统计 i→j 配对数
    count_i_to_j = 0
    count_j_to_i = 0

    for views in sample_view_clusters:  # views: shape (3,)
        # 属于 cluster i 的视图索引
        i_views = np.where(views == i)[0]
        j_views = np.where(views == j)[0]

        # 若某个视图在 i，其它视图在 j，则记录
        if len(i_views) > 0 and len(j_views) > 0:
            count_i_to_j += 1

        # 对称方向统计（可以和上面合并）
        if len(j_views) > 0 and len(i_views) > 0:
            count_j_to_i += 1

    isum = len(np.where(cluster_labels_ori == i)[0])
    jsum = len(np.where(cluster_labels_ori == j)[0])
    return count_i_to_j*1.0/isum, count_j_to_i*1.0/jsum

def merge_tensors(n, m, tensor1, tensor2, tensor3, tensor4):
    # 创建一个大小为 (n+m) * (n+m) 的零张量
    merged_tensor = torch.zeros((n + m, n + m))

    # 填充第一个 tensor 张量
    merged_tensor[:n, :n] = tensor1

    # 填充第二个 tensor 张量
    merged_tensor[:n, n:n + m] = tensor2

    # 填充第三个 tensor 张量
    merged_tensor[n:n + m, :n] = tensor3

    # 填充第四个 tensor 张量
    merged_tensor[n:n + m, n:n + m] = tensor4

    return merged_tensor

from pathlib import Path
import time

class Step():
    def __init__(self):
        self.step = 0
        self.round = {}
    def clear(self):
        self.step = 0
        self.round = {}
    def forward(self, x):
        self.step += x
    def reach_cycle(self, mod, ignore_zero = True):
        now = self.step // mod
        if now==0 and ignore_zero:
            return False
        if mod not in self.round or self.round[mod]!=now: #新过了一个或多个cycle
            self.round[mod] = now
            return True
        return False
    def state_dict(self):
        return {'step': self.step, 'round':self.round}
    def load_state_dict(self, state):
        self.step = state['step']
        self.round = state['round']
    @property
    def value(self):
        return self.step

class Logger():
    def __init__(self, file_name, mode = 'w', buffer = 100):
        (Path(file_name).parent).mkdir(exist_ok = True, parents = True)
        self.file_name = file_name
        self.fp = open(file_name, mode)
        self.cnt = 0
        self.stamp = time.time()
        self.buffer = buffer
    def log(self, *args, end='\n'):
        for x in args:
            if isinstance(x, dict):
                for y in x:
                    self.fp.write(str(y)+':'+str(x[y])+' ')
            else:
                self.fp.write(str(x)+' ')
        self.fp.write(end)
        self.cnt += 1
        if self.cnt>=self.buffer or time.time()-self.stamp>5:
            self.cnt = 0
            self.stamp = time.time()
            self.fp.close()
            self.fp = open(self.file_name, 'a')
        pass
    def close(self):
        self.fp.close()
