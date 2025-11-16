from sklearn.metrics import adjusted_rand_score, accuracy_score, normalized_mutual_info_score
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.optimize import linear_sum_assignment
import numpy as np


import torch
from torch.utils.data import DataLoader


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    # v_measure = v_measure_score(label, pred)
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur

def inference(loader, model, device, view, data_size):
    model.eval()
    pred_vectors = []
    Xs = []
    Zs = []
    Hs = []
    Qs = []
    LHZs = []
    HHZs = []
    for v in range(view):
        pred_vectors.append([])
        Xs.append([])
        Zs.append([])
        Hs.append([])
        Qs.append([])
        LHZs.append([])
        HHZs.append([])
    labels_vector = []

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            # zs, _, _, hs, _ = model.forward(xs)
            zs, lHZs, hHZs, xLNs, xLPs, xHPs, hs, zNoises, zPrivates = model(xs)
        for v in range(view):
            zs[v] = zs[v].detach()
            hs[v] = hs[v].detach()
            lHZs[v] = lHZs[v].detach()
            hHZs[v] = hHZs[v].detach()

            Xs[v].extend(xs[v].cpu().detach().numpy())
            Zs[v].extend(zs[v].cpu().detach().numpy())
            Hs[v].extend(hs[v].cpu().detach().numpy())
            LHZs[v].extend(lHZs[v].cpu().detach().numpy())
            HHZs[v].extend(hHZs[v].cpu().detach().numpy())

        labels_vector.extend(y.numpy())

    labels_vector = np.array(labels_vector).reshape(data_size)
    for v in range(view):
        Xs[v] = np.array(Xs[v])
        Zs[v] = np.array(Zs[v])
        Hs[v] = np.array(Hs[v])
        Qs[v] = np.array(Qs[v])
        LHZs[v] = np.array(LHZs[v])
        HHZs[v] = np.array(HHZs[v])
        pred_vectors[v] = np.array(pred_vectors[v])
    return Xs, pred_vectors, Zs, labels_vector, Hs, LHZs, HHZs

def valid(model, device, dataset, view, data_size, class_num, eval_h=True, eval_z=True, test=True, eval_lhz=True, eval_hhz = True, max_records=None):
    test_loader = DataLoader(dataset, batch_size=256, shuffle=False)
    X_vectors, pred_vectors, z_vectors, labels_vector, h_vectors, lhz_vectors, hhz_vectors = inference(test_loader, model, device, view, data_size)
    final_z_features = []
    h_clusters = []
    z_clusters = []
    saveFlag = False
    # maxAcc = max(max_records[0][0], max_records[1][0])
    # maxAcc = max(maxAcc, max_records[2][0])
    maxAcc = max_records[0][0]
    resFea = None

    if eval_h:
        acc_avg, nmi_avg, ari_avg, pur_avg = 0, 0, 0, 0
        for v in range(view):
            kmeans = KMeans(n_clusters=int(class_num), n_init=100)
            if len(labels_vector) > 10000:
                kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
            y_pred = kmeans.fit_predict(h_vectors[v])
            h_clusters.append(y_pred)
            nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
            acc_avg += acc
            nmi_avg += nmi
            ari_avg += ari
            pur_avg += pur
            if(max_records[1][0] < acc):
                max_records[1][0] = acc
                max_records[1][1] = nmi
                max_records[1][2] = ari
                max_records[1][3] = pur
                # if(acc > maxAcc):
                #     maxAcc = acc
                #     saveFlag = True

        if(max_records[2][0] < acc_avg / view):
            max_records[2][0] = acc_avg / view
            max_records[2][1] = nmi_avg / view
            max_records[2][2] = ari_avg / view
            max_records[2][3] = pur_avg / view

        kmeans = KMeans(n_clusters=class_num, n_init=100)
        if len(labels_vector) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        z = np.concatenate(h_vectors, axis=1)
        pseudo_label = kmeans.fit_predict(z)
        nmi, ari, acc, pur = evaluate(labels_vector, pseudo_label)
        if (max_records[0][0] < acc):
            max_records[0][0] = acc
            max_records[0][1] = nmi
            max_records[0][2] = ari
            max_records[0][3] = pur
            resFea = z
            saveFlag = True
            if (acc > maxAcc):
                maxAcc = acc
            #     saveFlag = True

    if eval_lhz:
        acc_avg, nmi_avg, ari_avg, pur_avg = 0, 0, 0, 0
        for v in range(view):
            kmeans = KMeans(n_clusters=int(class_num), n_init=100)
            if len(labels_vector) > 10000:
                kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
            y_pred = kmeans.fit_predict(lhz_vectors[v])
            h_clusters.append(y_pred)
            nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
            acc_avg += acc
            nmi_avg += nmi
            ari_avg += ari
            pur_avg += pur

            if (max_records[1][0] < acc):
                max_records[1][0] = acc
                max_records[1][1] = nmi
                max_records[1][2] = ari
                max_records[1][3] = pur
                # if (acc > maxAcc):
                #     maxAcc = acc
                #     saveFlag = True

        if (max_records[2][0] < acc_avg / view):
            max_records[2][0] = acc_avg / view
            max_records[2][1] = nmi_avg / view
            max_records[2][2] = ari_avg / view
            max_records[2][3] = pur_avg / view

        kmeans = KMeans(n_clusters=class_num, n_init=100)
        if len(labels_vector) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        z = np.concatenate(lhz_vectors, axis=1)
        pseudo_label = kmeans.fit_predict(z)
        nmi, ari, acc, pur = evaluate(labels_vector, pseudo_label)
        if (max_records[0][0] < acc):
            max_records[0][0] = acc
            max_records[0][1] = nmi
            max_records[0][2] = ari
            max_records[0][3] = pur
            resFea = z
            saveFlag = True
            if (acc > maxAcc):
                maxAcc = acc
                # saveFlag = True

    if eval_hhz:
        acc_avg, nmi_avg, ari_avg, pur_avg = 0, 0, 0, 0
        for v in range(view):
            kmeans = KMeans(n_clusters=int(class_num), n_init=100)
            if len(labels_vector) > 10000:
                kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
            y_pred = kmeans.fit_predict(hhz_vectors[v])
            h_clusters.append(y_pred)
            nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
            acc_avg += acc
            nmi_avg += nmi
            ari_avg += ari
            pur_avg += pur

            if (max_records[1][0] < acc):
                max_records[1][0] = acc
                max_records[1][1] = nmi
                max_records[1][2] = ari
                max_records[1][3] = pur
                # if (acc > maxAcc):
                #     maxAcc = acc
                #     saveFlag = True

        if (max_records[2][0] < acc_avg / view):
            max_records[2][0] = acc_avg / view
            max_records[2][1] = nmi_avg / view
            max_records[2][2] = ari_avg / view
            max_records[2][3] = pur_avg / view

        kmeans = KMeans(n_clusters=class_num, n_init=100)
        if len(labels_vector) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        z = np.concatenate(hhz_vectors, axis=1)
        pseudo_label = kmeans.fit_predict(z)
        nmi, ari, acc, pur = evaluate(labels_vector, pseudo_label)
        if (max_records[0][0] < acc):
            max_records[0][0] = acc
            max_records[0][1] = nmi
            max_records[0][2] = ari
            max_records[0][3] = pur
            resFea = z
            saveFlag = True
            if (acc > maxAcc):
                maxAcc = acc
                # saveFlag = True

    if eval_z:
        acc_avg, nmi_avg, ari_avg, pur_avg = 0, 0, 0, 0
        for v in range(view):
            kmeans = KMeans(n_clusters=int(class_num), n_init=100)
            if len(labels_vector) > 10000:
                kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
            y_pred = kmeans.fit_predict(z_vectors[v])
            final_z_features.append(z_vectors[v])
            z_clusters.append(y_pred)
            nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
            acc_avg += acc
            nmi_avg += nmi
            ari_avg += ari
            pur_avg += pur

            if (max_records[1][0] < acc):
                max_records[1][0] = acc
                max_records[1][1] = nmi
                max_records[1][2] = ari
                max_records[1][3] = pur
                # if (acc > maxAcc):
                #     maxAcc = acc
                #     saveFlag = True

        if (max_records[2][0] < acc_avg / view):
            max_records[2][0] = acc_avg / view
            max_records[2][1] = nmi_avg / view
            max_records[2][2] = ari_avg / view
            max_records[2][3] = pur_avg / view

        kmeans = KMeans(n_clusters=class_num, n_init=100)
        if len(labels_vector) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        h = np.concatenate(final_z_features, axis=1)
        pseudo_label = kmeans.fit_predict(h)
        nmi, ari, acc, pur = evaluate(labels_vector, pseudo_label)
        if (max_records[0][0] < acc):
            max_records[0][0] = acc
            max_records[0][1] = nmi
            max_records[0][2] = ari
            max_records[0][3] = pur
            resFea = h
            saveFlag = True
            if (acc > maxAcc):
                maxAcc = acc
                # saveFlag = True

    if test:
        kmeans = KMeans(n_clusters=class_num, n_init=100)
        if len(labels_vector) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=int(class_num), batch_size=5000, n_init=100)
        h = np.concatenate(z_vectors, axis=1)
        pseudo_label = kmeans.fit_predict(h)
        nmi, ari, acc, pur = evaluate(labels_vector, pseudo_label)
        if (max_records[0][0] < acc):
            max_records[0][0] = acc
            max_records[0][1] = nmi
            max_records[0][2] = ari
            max_records[0][3] = pur
            # saveFlag = True
            if (acc > maxAcc):
                maxAcc = acc
                # saveFlag = True
    print('ACC = {:.4f} PUR={:.4f}'.format(max_records[0][0], max_records[0][3]))

    return acc, pur, saveFlag, max_records, resFea, labels_vector
