from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')
        print(self.data1.shape)
        print(self.data2.shape)
        print(self.data3.shape)
        # scipy.io.savemat('CCV.mat', {'X1': self.data1, 'X2': self.data2, 'X3': self.data3, 'Y': self.labels})

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class Caltech_6V(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)        
        scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
        # for i in [0, 3]:
            self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class NUSWIDE(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        # scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        # print(self.class_num)
        # for i in range(5000):
        #     print(data['X1'][i][-1])
        # X1 = data['X1'][:, :-1]
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)][:, :-1].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)][:, :-1].shape)
            self.dims.append(data['X' + str(i + 1)][:, :-1].shape[1])
        self.data_size = self.multi_view[0].shape[0]
        print(self.labels.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class DHA(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class CoRA(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y']
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])
        self.data_size = self.multi_view[0].shape[0]
        print(self.labels.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class MNIST(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            temp = data['X' + str(i + 1)].reshape((data['X' + str(i + 1)].shape[0], -1))
            self.multi_view.append(temp.astype(np.float32))
            print(temp.shape)
            self.dims.append(temp.shape[1])
        self.data_size = self.multi_view[0].shape[0]
        print(self.labels.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class MNIST_USPS(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y']
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            temp = data['X' + str(i + 1)].reshape((data['X' + str(i + 1)].shape[0], -1))
            self.multi_view.append(temp.astype(np.float32))
            print(temp.shape)
            self.dims.append(temp.shape[1])
        self.data_size = self.multi_view[0].shape[0]
        print(self.labels.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class MultiCOIL(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(view):
            temp = data['X' + str(i + 1)].reshape((data['X' + str(i + 1)].shape[0], -1))
            self.multi_view.append(temp.astype(np.float32))
            print(temp.shape)
            self.dims.append(temp.shape[1])
        self.data_size = self.multi_view[0].shape[0]
        print(self.labels.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class OthersDataSet(Dataset):
    def __init__(self, path, dataset = None):
        raw = scipy.io.loadmat(path)
        allData = raw['fea']
        # print(data)
        self.view = allData.shape[1]
        self.multi_view = []
        self.labels = raw['gt']
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(self.view):
            if dataset in ['UCI_Digits','3Sources','MSRC_v1','NUS_WIDE']:
                temp = allData[0, i]
            else:
                temp = allData[0, i].toarray()
            self.multi_view.append(temp.astype(np.float32))
            print(temp.shape)
            self.dims.append(temp.shape[1])
        self.data_size = self.multi_view[0].shape[0]
        print(self.labels.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class OthersDataSetTwo(Dataset):
    def __init__(self, path, dataset = None):
        raw = scipy.io.loadmat(path)
        allData = raw['X']
        # print(data)
        self.view = allData.shape[0]
        self.multi_view = []
        self.labels = raw['y']
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        for i in range(self.view):
            temp = allData[i, :][0]
            self.multi_view.append(temp.astype(np.float32))
            print(temp.shape)
            self.dims.append(temp.shape[1])
        self.data_size = self.multi_view[0].shape[0]
        print(self.labels.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class YoutubeVideo(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        # print(data)
        scaler = MinMaxScaler()
        self.view = view
        self.multi_view = []
        self.labels = data['Y'].T-1
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(3):
            if(i==1):
                continue
            self.multi_view.append(data['X' + str(i + 1)].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data['X' + str(i + 1)].shape)
            self.dims.append(data['X' + str(i + 1)].shape[1])

        self.data_size = self.multi_view[0].shape[0]
        print(self.labels.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

import hdf5storage as hdf

def load_mat(path, views=None, key_feature="data", key_label="labels"):
    data = hdf.loadmat(path)
    feature = []
    num_view = len(data[key_feature])
    label = data[key_label].reshape((-1,))
    num_smp = label.size
    for v in range(num_view):
        tmp = data[key_feature][v][0].squeeze()
        feature.append(tmp)
    # 打乱样本
    rand_permute = np.random.permutation(num_smp)
    for v in range(num_view):
        feature[v] = feature[v][rand_permute]
    label = label[rand_permute]
    if views is None or len(views) == 0:
        views = list(range(num_view))
    views_feature = [feature[v] for v in views]
    return views_feature, label

class Caltech101(Dataset):
    def __init__(self, path, view=None):
        data, labels = load_mat(path)
        # print(data)
        # scaler = MinMaxScaler()
        if(view is None):
            self.view = len(data)
        else:
            self.view = view
        self.multi_view = []
        self.labels = labels.reshape((-1,1))
        self.dims = []
        self.class_num = len(np.unique(self.labels))
        print(self.class_num)
        for i in range(view):
            self.multi_view.append(data[i].astype(np.float32))
            # self.multi_view.append(scaler.fit_transform(data['X' + str(i + 1)].astype(np.float32)))
            print(data[i].shape)
            self.dims.append(data[i].shape[1])

        self.data_size = self.multi_view[0].shape[0]
        print(self.labels.shape)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        data_getitem = []
        for i in range(self.view):
            data_getitem.append(torch.from_numpy(self.multi_view[i][idx]))
        return data_getitem, torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class MultiviewDataset(Dataset):
    def __init__(self, root: str, device,
                 key_feature="data", key_label="labels",
                 normalize=True, views=None):
        """
        :param root: 数据集路径
        :param device: cpu or cuda
        :param key_feature: 存储特征的关键字
        :param key_label: 存储标签的关键字
        :param views: 指定读取哪些视图
        """
        # load_scene --- customize load_<dataset> function if not consistent
        data, labels = load_mat(root, views, key_feature=key_feature, key_label=key_label)
        num_view = len(data)
        view_dims = [0] * num_view
        for i in range(num_view):
            # .to(device)
            data[i] = torch.as_tensor(data[i], dtype=torch.float32).to(device)
            if normalize:
                max_value, _ = torch.max(data[i], dim=0, keepdim=True)
                min_value, _ = torch.min(data[i], dim=0, keepdim=True)
                data[i] = (data[i] - min_value) / (max_value - min_value + 1e-12)
                # data[i] = (data[i] - torch.min(data[i])) / (torch.max(data[i]) - torch.min(data[i]) + 1e-12)
            view_dims[i] = data[i].shape[1]
        self.data = data
        self.num_view = num_view
        self.view_dims = view_dims
        # .to(device)
        self.labels = torch.as_tensor(labels, dtype=torch.int32).view((-1,)).to(device)
        self.num_class = len(torch.unique(self.labels))
        self.device = device

    def __getitem__(self, index):
        item = []
        for i in range(self.num_view):
            # .to(self.device)
            item.append(self.data[i][index])
        # .to(self.device)
        return item, torch.from_numpy(self.labels[index]), torch.from_numpy(np.array(index)).long()

    def __len__(self):
        return len(self.labels)
    

import os

def load_data(dataset):
    # basePath = r"D:\Documents\Python\myDriveData\MultiViews\matData"
    # basePath = r"D:\Documents\Python\myDriveData\MultiViews\Multi-view-datasets-master"
    # basePath = "/home/zwt/projects/lsy/code20250507/data"
    basePath = "./data"
    if dataset == "CCV":
        dataset = CCV(basePath+"/")
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Caltech":
        dataset = Caltech_6V(os.path.join(basePath, 'Caltech.mat'), view=6)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE(os.path.join(basePath,'NUSWIDE.mat'), view=5)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "DHA":
        dataset = DHA(os.path.join(basePath, 'DHA.mat'), view=2)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "YoutubeVideo":
        dataset = YoutubeVideo(os.path.join(basePath, "Video-3V.mat"), view=2)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "CoRA":
        dataset = CoRA(os.path.join(basePath, "cora_4V_noNormalize.mat"), view=4)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset in ["ALOI_100", "BBCSport_Two", "Reuters_21578", "UCI_Digits", "Reuters", "CiteSeer", "3Sources", "MSRC_v1", "NUS_WIDE"]:
        dataset = OthersDataSet(os.path.join(basePath, f"{dataset}.mat"), dataset=dataset)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset == "Multi-FMNIST":
        dataset = MNIST(os.path.join(basePath, f"{dataset}.mat"), view=3)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset in ["Cora", "ALOI", "BBCSport", "ORL", "NUS-WIDE", "WebKB", "Prokaryotic", "MNIST-4", "MNIST-10k", "Caltech101-7", "Caltech101-20", "Caltech101-all", "Reuters-1200"]:
        dataset = OthersDataSetTwo(os.path.join(basePath, f"{dataset}.mat"), dataset=dataset)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    elif dataset in ["Multi-COIL-20", "Multi-COIL-10"]: # views=3 3
        dataset = MultiCOIL(os.path.join(basePath, f"{dataset}.mat"), view=3)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    # elif dataset == "Caltech101-20":
    #     dataset = Caltech101(os.path.join(basePath, f"{dataset}.mat"),view=6)
    #     dims = dataset.dims
    #     view = dataset.view
    #     data_size = dataset.data_size
    #     class_num = dataset.class_num
    elif dataset == "MNIST_USPS":
        dataset = MNIST_USPS(os.path.join(basePath, f"mnist_usps_normalize.mat"),view=2)
        dims = dataset.dims
        view = dataset.view
        data_size = dataset.data_size
        class_num = dataset.class_num
    else:
        raise NotImplementedError
    print(f"dims: {dims}, view: {view}, data_size: {data_size}, classnum: {class_num}")
    return dataset, dims, view, data_size, class_num
