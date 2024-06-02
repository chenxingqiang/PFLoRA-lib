import numpy as np
import os
import sys
import random
import torch
from datasets import load_dataset
from utils.dataset_utils import check, separate_data, split_data, save_file

random.seed(1)
np.random.seed(1)

num_clients = 20
dir_path = "GLUE/"


def generate_dataset(dir_path, num_clients, niid, balance, partition, task_name):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Load GLUE dataset
    dataset = load_dataset("glue", task_name)
    train_data = dataset["train"]
    test_data = dataset["validation"]

    # Preprocess data
    train_texts = train_data["sentence1"] if task_name != "stsb" else train_data["sentence1"] + \
        train_data["sentence2"]
    train_labels = train_data["label"]
    test_texts = test_data["sentence1"] if task_name != "stsb" else test_data["sentence1"] + \
        test_data["sentence2"]
    test_labels = test_data["label"]

    dataset_text = []
    dataset_label = []
    dataset_text.extend(train_texts)
    dataset_text.extend(test_texts)
    dataset_label.extend(train_labels)
    dataset_label.extend(test_labels)

    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    X, y, statistic = separate_data(
        (dataset_text, dataset_label), num_clients, num_classes, niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data,
              num_clients, num_classes, statistic, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    task_name = sys.argv[4]

    generate_dataset(dir_path, num_clients, niid,
                     balance, partition, task_name)
