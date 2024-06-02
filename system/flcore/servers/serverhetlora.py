# ./system/flcore/servers/serverhetlora.py

import os
import time
import numpy as np
import torch
import torch.nn as nn
from flcore.servers.serverbase import Server

from flcore.clients.clienthetlora import clientHetLoRA
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import concurrent.futures
import h5py


class serverHetLoRA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.rs_auc = []
        self.rs_test_acc = []
        self.updates = []
        self.rs_time_cost = []

        # Initialize test loader
        self.test_loader = self.load_test_data()

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientHetLoRA)
        self.loss = nn.CrossEntropyLoss()

        print(
            f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

    def train(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for i in range(self.global_rounds + 1):
                start_time = time.time()
                self.selected_clients = self.select_clients()
                self.send_models()

                if i % self.eval_gap == 0:
                    print(f"\n-------------Round number: {i}-------------")
                    print("\nEvaluate global model")
                    self.evaluate(self.global_model, self.test_loader)

                future_to_client = {executor.submit(
                    client.train, client.model): client for client in self.selected_clients}
                concurrent.futures.wait(future_to_client)

                self.receive_models()
                self.aggregate_parameters()
                end_time = time.time()
                self.rs_time_cost.append(end_time - start_time)

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.rs_time_cost[1:]) / len(self.rs_time_cost[1:]))

        self.save_results()
        self.save_global_model()

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.model.base_model.load_state_dict(
                self.global_model.base_model.state_dict())
            client.model.lora_layers = client.model.add_lora_layers()
            client.model.truncate_lora_layer(client.model)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:
            lora_state_dict = client.train(net=client.model)
            self.updates.append(lora_state_dict)

    def aggregate_parameters(self):
        assert (len(self.updates) > 0)
        agg_state_dict = self.global_model.state_dict()
        for key in agg_state_dict:
            if key.startswith('lora_layers.'):
                agg_state_dict[key] = torch.stack(
                    [update[key] for update in self.updates]).mean(dim=0)
        self.global_model.load_state_dict(agg_state_dict)
        self.updates = []

    def load_test_data(self):
        # Assuming the test data is stored with client id 0
        test_data = read_client_data(self.dataset, 0, is_train=False)
        return DataLoader(test_data, batch_size=self.args.batch_size, drop_last=False, shuffle=False)

    def evaluate(self, model, testloader):
        model.eval()
        test_acc = 0
        test_num = 0
        correct = 0
        total = 0
        total_loss = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for batch in testloader:
                if isinstance(batch, dict):
                    input_ids = batch['input_ids']
                    attention_mask = batch.get('attention_mask', None)
                    labels = batch['labels']
                else:
                    if len(batch) == 3:
                        input_ids, attention_mask, labels = batch
                    elif len(batch) == 2:
                        input_ids, labels = batch
                        attention_mask = None
                    else:
                        raise ValueError("Unexpected batch format")

                input_ids = input_ids.to(self.device, dtype=torch.long)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(
                        self.device, dtype=torch.long)
                labels = labels.to(self.device, dtype=torch.long)

                if attention_mask is not None:
                    output = model(input_ids=input_ids,
                                   attention_mask=attention_mask)
                else:
                    output = model(input_ids=input_ids)

                test_acc += (torch.sum(torch.argmax(output, dim=1)
                                       == labels)).item()
                test_num += labels.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(
                    labels.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

                logits = output
                loss = self.loss(logits, labels)
                total_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        average_loss = total_loss / total


        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        self.rs_test_acc.append(test_acc / test_num)
        self.rs_auc.append(auc)
        print(f"Test Accuracy: {test_acc / test_num}, AUC: {auc}")

        print(
            f"Evaluation - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy, average_loss

    def save_results(self):
        algo = self.algorithm
        result_path = os.path.join("..", "results", "GLUE")

        # 逐级创建文件夹路径
        os.makedirs(result_path, exist_ok=True)
        if len(self.rs_test_acc):
            algo = self.dataset.split(
                "/")[-1]+"_" + algo + "_" + self.goal + "_" + str(self.times)
            file_path = os.path.join(result_path, "{}.h5".format(algo))
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_time_cost', data=self.rs_time_cost)
                # Add any other results you need to save
