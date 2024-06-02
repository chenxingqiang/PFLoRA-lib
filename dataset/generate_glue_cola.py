import os
import numpy as np
import pandas as pd
import random
import sys
import json
import ujson
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

batch_size = 10
train_ratio = 0.75
alpha = 0.1  # for Dirichlet distribution. 100 for exdir
random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "GLUE/CoLA"


def preprocess(tokenizer, examples):
    if 'sentence2' in examples:
        texts = (examples["sentence1"], examples["sentence2"])
    else:
        texts = (examples["sentence"],)
    result = tokenizer(*texts, padding="max_length",
                      max_length=128, truncation=True)
    return result


def standardize_columns(file_path):
    df = pd.read_csv(file_path, delimiter='\t', header=None)
    expected_columns = {
        4: ['ignore', 'label', 'ignore2', 'sentence'],
        3: ['label', 'ignore', 'sentence'],
        2: ['label', 'sentence']
    }
    if df.shape[1] in expected_columns:
        df.columns = expected_columns[df.shape[1]]
    else:
        raise ValueError(
            f"Unexpected number of columns ({df.shape[1]}) in the TSV file.")
    return df[['label', 'sentence']]


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=None):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data
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
        C = class_per_client
        min_size_per_label = 0
        min_require_size_per_label = max(
            C * num_clients // num_classes // 2, 1)
        if min_require_size_per_label < 1:
            raise ValueError
        clientidx_map = {}
        while min_size_per_label < min_require_size_per_label:
            for k in range(num_classes):
                clientidx_map[k] = []
            for i in range(num_clients):
                labelidx = np.random.choice(
                    range(num_classes), C, replace=False)
                for k in labelidx:
                    clientidx_map[k].append(i)
            min_size_per_label = min([len(clientidx_map[k])
                                    for k in range(num_classes)])

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

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(
                    np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients and j in clientidx_map[k])
                                      for j, (p, idx_j) in enumerate(zip(proportions, idx_batch))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                              len(idx_k)).astype(int)[:-1]
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

    for client in range(num_clients):
        idxs = dataidx_map.get(client, [])
        if idxs:
            X[client] = dataset_content[idxs]
            y[client] = dataset_label[idxs]

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data
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

        train_data.append({
            'input_ids': X_train,
            # Assuming no attention mask was provided, using dummy ones
            'attention_mask': np.ones_like(X_train),
            'labels': y_train
        })
        num_samples['train'].append(len(y_train))

        test_data.append({
            'input_ids': X_test,
            # Assuming no attention mask was provided, using dummy ones
            'attention_mask': np.ones_like(X_test),
            'labels': y_test
        })
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(
        num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, statistic, niid=False, balance=True, partition=None):
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

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    print("Saving to disk.\n")

    for idx, train_dict in enumerate(train_data):
        with open(os.path.join(train_path, f'{idx}.npz'), 'wb') as f:
            np.savez_compressed(
                f, input_ids=train_dict['input_ids'], attention_mask=train_dict['attention_mask'], labels=train_dict['labels'])
    for idx, test_dict in enumerate(test_data):
        with open(os.path.join(test_path, f'{idx}.npz'), 'wb') as f:
            np.savez_compressed(
                f, input_ids=test_dict['input_ids'], attention_mask=test_dict['attention_mask'], labels=test_dict['labels'])
    with open(config_path, 'w') as f:
        json.dump(config, f)

    print("Finish generating dataset.\n")


def generate_dataset(dir_path, num_clients, niid, balance, partition, task_name):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    config_path = os.path.join(dir_path, "config.json")
    train_path = os.path.join(dir_path, "train")
    test_path = os.path.join(dir_path, "test")

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    dataset_path = os.path.join(os.path.expanduser(
        "~/datasets"), "glue", task_name.lower())

    train_df = standardize_columns(
        os.path.join(dataset_path, "in_domain_train.tsv"))
    validation_df = standardize_columns(
        os.path.join(dataset_path, "in_domain_dev.tsv"))
    test_df = standardize_columns(os.path.join(
        dataset_path, "out_of_domain_dev.tsv"))

    datasets = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'validation': Dataset.from_pandas(validation_df),
        'test': Dataset.from_pandas(test_df)
    })

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    datasets = datasets.map(lambda x: preprocess(tokenizer, x), batched=True)

    dataset_labels = np.array(datasets["train"]["label"])

    num_classes = len(set(dataset_labels))
    print(f'Number of classes: {num_classes}')

    X_train, y_train = np.array(datasets["train"]["input_ids"]), dataset_labels
    X, y, statistic = separate_data(
        (X_train, y_train), num_clients, num_classes, niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data,
             num_clients, num_classes, statistic, niid, balance, partition)


def check(config_path, train_path, test_path, num_clients, niid, balance, partition):
    if not os.path.exists(config_path):
        return False

    with open(config_path, 'r') as f:
        config = json.load(f)

    if config.get('num_clients') == num_clients and \
       config.get('non_iid') == niid and \
       config.get('balance') == balance and \
       config.get('partition') == partition and \
       os.path.exists(train_path) and os.path.exists(test_path):
        print("Dataset already generated with the same parameters.")
        return True

    return False


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    task_name = sys.argv[4]

    generate_dataset(dir_path, num_clients, niid,
                     balance, partition, task_name)
