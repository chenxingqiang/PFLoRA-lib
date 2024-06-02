# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import json
import os
import ujson
import numpy as np
import gc
from sklearn.model_selection import train_test_split

batch_size = 10
# merge original training set and test set, then split it manually.
train_ratio = 0.75
alpha = 0.1  # for Dirichlet distribution. 100 for exdir


def check(config_path, train_path, test_path, num_clients, niid=False,
          balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
                config['non_iid'] == niid and \
                config['balance'] == balance and \
                config['partition'] == partition and \
                config['alpha'] == alpha and \
                config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
    # guarantee that each client must have at least one batch of data for testing.
    least_samples = int(min(batch_size / (1-train_ratio),
                        len(dataset_label) / num_clients / 2))

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            idx_for_each_class.append(idxs[dataset_label == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(
                np.ceil((num_clients/num_classes)*class_per_client))]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per)
                               for _ in range(num_selected_clients-1)]
            else:
                num_samples = np.random.randint(max(
                    num_per/10, least_samples/num_classes), num_per, num_selected_clients-1).tolist()
            num_samples.append(num_all_samples-sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx+num_sample]
                else:
                    dataidx_map[client] = np.append(
                        dataidx_map[client], idx_for_each_class[i][idx:idx+num_sample], axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        try_cnt = 1
        while min_size < least_samples:
            if try_cnt > 1:
                print(
                    f'Client data size does not meet the minimum requirement {least_samples}. Try allocating again for the {try_cnt}-th time.')

            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(alpha, num_clients))
                proportions = np.array(
                    [p*(len(idx_j) < N/num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions/proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j,
                             idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
            try_cnt += 1

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

    elif partition == 'exdir':
        r'''This strategy comes from https://arxiv.org/abs/2311.03154
        See details in https://github.com/TsingZ0/PFLlib/issues/139

        This version in PFLlib is slightly different from the original version
        Some changes are as follows:
        n_nets -> num_clients, n_class -> num_classes
        '''
        C = class_per_client

        '''The first level: allocate labels to clients
        clientidx_map (dict, {label: clientidx}), e.g., C=2, num_clients=5, num_classes=10
            {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5],
                5: [5, 6], 6: [6, 7], 7: [7, 8], 8: [8, 9], 9: [9, 0]}
        '''
        min_size_per_label = 0
        # You can adjust the `min_require_size_per_label` to meet you requirements
        min_require_size_per_label = max(
            C * num_clients // num_classes // 2, 1)
        if min_require_size_per_label < 1:
            raise ValueError
        clientidx_map = {}
        while min_size_per_label < min_require_size_per_label:
            # initialize
            for k in range(num_classes):
                clientidx_map[k] = []
            # allocate
            for i in range(num_clients):
                labelidx = np.random.choice(
                    range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k])
                                     for k in range(num_classes)])

        '''The second level: allocate data idx'''
        dataidx_map = {}
        y_train = dataset_label
        min_size = 0
        min_require_size = 10
        K = num_classes
        N = len(y_train)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[i]) for i in range(len(clientidx_map))])

        # ensure per client' sampling size >= min_require_size (is set to 10 originally in [3])
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(alpha, num_clients))
                # Balance
                # Case 1 (original case in Dir): Balance the number of sample per client
                proportions = np.array([p * (len(idx_j) < N / num_clients and j in clientidx_map[k])
                                       for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                # Case 2: Don't balance
                # proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                # process the remainder samples
                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], num_clients-1):
                        proportions[w] = len(idx_k)
                idx_batch = [idx_j + idx.tolist() for idx_j,
                             idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            np.random.shuffle(idx_batch[j])
            dataidx_map[j] = idx_batch[j]

    elif partition == "":
        # Randomly partition the data across clients
        for i, (x, y_value) in enumerate(zip(dataset_content, dataset_label)):
            client_idx = i % num_clients
            if client_idx not in dataidx_map:
                dataidx_map[client_idx] = []
            dataidx_map[client_idx].append(i)

        for client in range(num_clients):
            idxs = dataidx_map.get(client, [])
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]
            for label in np.unique(y[client]):
                statistic[client].append(
                    (int(label), int(sum(y[client] == label))))
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map.get(client, [])
        if idxs:
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(
            y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)
    return X, y, statistic



def split_data(X, y):
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_ratio, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(
        num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data

    return False


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients,
              num_classes, statistic, niid=False, balance=True, partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # Create directories if they do not exist
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(os.path.join(train_path, f'{idx}.npz'), 'wb') as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(os.path.join(test_path, f'{idx}.npz'), 'wb') as f:
            np.savez_compressed(f, data=test_dict)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")

def check(config_path, train_path, test_path, num_clients, niid, balance, partition):
    # Check if the configuration file exists
    if not os.path.exists(config_path):
        return False

    # Load the configuration file
    with open(config_path, 'r') as f:
        config = ujson.load(f)

    # Compare the parameters with those stored in the configuration file
    if config.get('num_clients') == num_clients and \
       config.get('non_iid') == niid and \
       config.get('balance') == balance and \
       config.get('partition') == partition and \
       os.path.exists(train_path) and os.path.exists(test_path):
        print("Dataset already generated with the same parameters.")
        return True

    return False
