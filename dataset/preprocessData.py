import pandas as pd
import os
import numpy as np
import scipy.io as scio
from sklearn.preprocessing import LabelEncoder

root_path = "" # r"D:\Documents\Python\myDriveData\MultiViews\cora"
save_path = "" # r"D:\Documents\Python\myDriveData\MultiViews\matData"

def saveCoRA():
    raw_data = pd.read_csv(os.path.join(root_path, 'cora.content'), sep='\t', header=None)
    print("content shape: ", raw_data.shape)

    features = raw_data.iloc[:, 1:-1]
    print("features shape: ", features.shape)
    print(type(features))

    le = LabelEncoder()
    raw_data[1434] = le.fit_transform(raw_data[1434])
    # one-hot encoding
    onehot_labels = pd.get_dummies(raw_data[1434])
    labels = raw_data[1434].values.reshape((-1,1))
    print(type(onehot_labels))
    print("\n----labels----")
    print(type(labels))
    ySet = set(list(raw_data[1434].values))
    print(ySet)
    yNum = len(ySet)
    views = 4
    print(f"label kind num: {yNum}")

    X1 = None
    X2 = None
    X3 = None
    X4 = None
    Y = []

    for yy in range(yNum):
        temp = raw_data[raw_data[1434]==yy]
        features = temp.iloc[:, 1:-1].values.reshape((-1, 1433))
        vLen = features.shape[0] // views

        XX1 = features[:vLen, :]
        XX2 = features[vLen:vLen*2, :]
        XX3 = features[vLen*2:vLen*3, :]
        XX4 = features[vLen*3:vLen*4, :]

        assert XX1.shape[0] == XX2.shape[0] and XX3.shape[0] == XX2.shape[0] and XX3.shape[0] == XX4.shape[0], f"XX1: {XX1.shape}, XX3: {XX2.shape}, XX3: {XX3.shape}, XX4: {XX4.shape}"
        if(X1 is None):
            X1 = XX1
            X2 = XX2
            X3 = XX3
            X4 = XX4
        else:
            X1 = np.concatenate((X1, XX1))
            X2 = np.concatenate((X2, XX2))
            X3 = np.concatenate((X3, XX3))
            X4 = np.concatenate((X4, XX4))
        Y.extend([yy] * vLen)
        assert X1.shape[0] == X2.shape[0] and X3.shape[0] == X2.shape[0] and X3.shape[0] == X4.shape[0] and X3.shape[0] == len(Y), f"X1: {X1.shape}, X2: {X2.shape}, X3: {X3.shape}, X4: {X4.shape}, Y: {len(Y)}"
        print(f"X1: {X1.shape}, X2: {X2.shape}, X3: {X3.shape}, X4: {X4.shape}, Y: {len(Y)}")
    Y = np.array(Y).reshape((-1,1))

    # from sklearn import preprocessing
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X1 = min_max_scaler.fit_transform(X1)
    # X2 = min_max_scaler.fit_transform(X2)
    # X3 = min_max_scaler.fit_transform(X3)
    # X4 = min_max_scaler.fit_transform(X4)

    scio.savemat(save_path + '/cora_4V_noNormalize.mat', {'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'Y': Y})

def scene15():
    raw_data = scio.loadmat(os.path.join(save_path, 'scene15.mat'))
    data = raw_data['X1'][0, :]
    y = list(raw_data['y'][0, :])

    X1 = None
    X2 = None
    X3 = None
    Y = []
    views = 3
    begin = 0
    end = 0
    w= 100
    h= 100

    for yy in range(15):
        begin = end
        i = begin
        while(i <len(y) and y[i]==yy):
            i+=1
        end = i

        num = end-begin
        vLen = num // views
        for j in range(begin, begin+vLen):
            # temp1 = data[j].reshape(1, data[j].shape[0], data[j].shape[1], 1)
            # temp2 = data[j+vLen].reshape(1, data[j+vLen].shape[0], data[j+vLen].shape[1], 1)
            # temp3 = data[j+vLen*2].reshape(1, data[j+vLen*2].shape[0], data[j+vLen*2].shape[1], 1)
            temp11 = data[j].reshape(1, data[j].shape[0], data[j].shape[1], 1)
            temp22 = data[j + vLen].reshape(1, data[j + vLen].shape[0], data[j + vLen].shape[1], 1)
            temp33 = data[j + vLen * 2].reshape(1, data[j + vLen * 2].shape[0], data[j + vLen * 2].shape[1], 1)
            temp1 = np.resize(temp11, (1, w, h, 1))
            temp2 = np.resize(temp22, (1, w, h, 1))
            temp3 = np.resize(temp33, (1, w, h, 1))
            if(X1 is None):
                X1 = temp1
                X2 = temp2
                X3 = temp3
            else:
                # print(f"X1: {X1.shape}, temp1: {temp1.shape}")
                # print(f"X2: {X2.shape}, temp2: {temp2.shape}")
                # print(f"X3: {X3.shape}, temp3: {temp3.shape}")
                X1 = np.concatenate((X1, temp1))
                X2 = np.concatenate((X2, temp2))
                X3 = np.concatenate((X3, temp3))

        Y.extend([yy] * vLen)
        assert X1.shape[0] == X2.shape[0] and X3.shape[0] == X2.shape[0] and X3.shape[
            0] == len(Y), f"X1: {X1.shape}, X2: {X2.shape}, X3: {X3.shape}, Y: {len(Y)}"

    X1 = X1/255.0
    X2 = X2/255.0
    X3 = X3/255.0
    scio.savemat(save_path + '/scence15_3V.mat', {'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y})

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
def mnist_usps():
    raw_data1 = scio.loadmat(os.path.join(save_path, 'mnist_train.mat'))
    labels1 = scio.loadmat(os.path.join(save_path, 'mnist_train_labels.mat'))
    labels2 = scio.loadmat(os.path.join(save_path, 'usps_train_labels.mat'))
    raw_data2 = scio.loadmat(os.path.join(save_path, 'usps_train.mat'))
    data1 = raw_data1['mnist_train']
    data2 = raw_data2['usps_train']
    l2 = labels2['usps_train_labels'].T
    l2 = list(l2[0, :])
    yDF2 = pd.DataFrame({'label': l2, 'ind': [i for i in range(len(l2))]})
    yDF2['label'] = yDF2['label']-1
    l1 = labels1['mnist_train_labels'].T
    l1 = list(l1[0, :])
    yDF1 = pd.DataFrame({'label': l1, 'ind': [i for i in range(len(l1))]})

    X1 = None
    X2 = None
    Y= []

    for yy in range(10):
        tem2 = yDF2[yDF2['label']==yy]
        xNum = tem2.shape[0]
        tem1 = yDF1[yDF1['label']==yy]
        tem1 = tem1.iloc[:xNum]

        ind1 = tem1['ind'].tolist()
        ind2 = tem2['ind'].tolist()
        XX1 = data1[ind1]
        XX2 = data2[ind2]
        assert XX1.shape[0] == XX2.shape[0], f"XX1: {XX1.shape}, XX3: {XX2.shape}"
        if (X1 is None):
            X1 = XX1
            X2 = XX2

        else:
            X1 = np.concatenate((X1, XX1))
            X2 = np.concatenate((X2, XX2))

        Y.extend([yy] * xNum)
        assert X1.shape[0] == X2.shape[0] and X2.shape[0] == len(Y), f"X1: {X1.shape}, X2: {X2.shape}, Y: {len(Y)}"
        print(f"X1: {X1.shape}, X2: {X2.shape}, Y: {len(Y)}")
    Y = np.array(Y).reshape((-1, 1))
    X1 = X1/255.0
    X2 = X2/255.0
    scio.savemat(save_path + '/mnist_usps_normalize.mat', {'X1': X1, 'X2': X2, 'Y': Y})

data_path_two = r"D:\Documents\Python\myDriveData\MultiViews\Multi-view-datasets-master"
# raw_data = scio.loadmat(os.path.join(data_path_two, 'ORL.mat'))
raw_data = scio.loadmat(os.path.join(data_path_two, 'BBC4view.mat'))

# import h5py
# file_path = os.path.join(data_path_two, 'BBC4view.mat')
# with h5py.File(file_path, 'r') as f:
    # # 查看所有键（变量名）
    # print("Available keys in .mat file:", list(f.keys()))

# views_feature, label = load_mat(os.path.join(save_path, 'Caltech101-20.mat'))

# data = np.load(r"D:\Documents\Python\myDriveData\noiseMNIST\noise_train.npy", allow_pickle=True)
# threshData = np.load(r"D:\Documents\Python\myDriveData\noiseMNIST\thresh_train.npy", allow_pickle=True)
# csv = pd.read_csv(r"D:\Documents\Python\myDriveData\noiseMNIST\noisy_mnist.csv")