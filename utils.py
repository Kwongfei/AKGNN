import sys

import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import random 

import torch

import os


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(dataset_str, seed, nodes_per_class = 20):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = normalize(features)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # 检测到adj不对称的时候，取最大值。即若 a_ij != a_ji, 则 a_ij=a_ji=max(a_ij, a_ji)
    A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)    


    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]     # onehot

    idx_val = range(len(y), len(y)+500)
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.argmax(labels, -1))
    A = sparse_mx_to_torch_sparse_tensor(adj)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, features, labels, idx_train, idx_val, idx_test

def load_new_data(dataset_name, train_label_rate, val_label_rate, random_seed=123, random_split=False):
    dataset_folder = '../GGCN/new_data'

    dataset_path = os.path.join(dataset_folder, dataset_name)

    # 导入连接的点对信息
    edge_file = os.path.join(dataset_path, 'out1_graph_edges.txt')
    edges = np.loadtxt(edge_file, skiprows=1, dtype=int)
    num_nodes = np.max(edges) + 1

    # 创建稀疏矩阵
    adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                     shape=(num_nodes, num_nodes))

    # 检测到adj不对称的时候，取最大值。即若 a_ij != a_ji, 则 a_ij=a_ji=max(a_ij, a_ji)
    A = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # 导入节点特征和标签
    feature_file = os.path.join(dataset_path, 'out1_node_feature_label.txt')
    data = np.loadtxt(feature_file, skiprows=1, dtype=str)

    # 处理节点特征
    node_ids = data[:, 0].astype(int)
    if dataset_name == 'film':
        features = [list(map(lambda x: int(x) - 1, row.split(','))) for row in data[:, 1]]
    else:
        features = np.array([list(map(lambda x: int(x), row.split(','))) for row in data[:, 1]])
    labels = data[:, 2].astype(int)

    if dataset_name == 'film':
        # 创建空的特征矩阵
        feature_amount = 931  # 特征的总数量
        feature_matrix = np.zeros((len(data), feature_amount), dtype=int)

        # 填充特征矩阵中对应索引位置为1
        for i, feature_indices in enumerate(features):
            feature_matrix[i, feature_indices] = 1
        features = feature_matrix

    # 根据节点编号排序特征和标签
    sorted_indices = np.argsort(node_ids)
    node_ids = node_ids[sorted_indices]
    features = features[sorted_indices]
    labels = labels[sorted_indices]

    # 根据划分比例计算样本数量
    num_samples = len(labels)
    num_classes = len(np.unique(labels))
    num_train = int(num_samples * train_label_rate)
    percls_trn = int(round(num_samples * train_label_rate/num_classes))
    num_val = int(num_samples * val_label_rate)
    num_test = num_samples - num_train - num_val
    index = [i for i in range(0, num_samples)]

    train_idx = []
    rnd_state = np.random.RandomState(random_seed)
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, num_val, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    A = sparse_mx_to_torch_sparse_tensor(adj)
    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)

    return A, features, labels, train_idx, val_idx, test_idx

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
