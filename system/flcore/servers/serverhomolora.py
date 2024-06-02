# ./system/flcore/servers/serverhomolora.py

import numpy as np
from sklearn import metrics
from sklearn.calibration import label_binarize
import torch
import torch.nn as nn
from flcore.servers.serverbase import Server
from flcore.trainmodel.homo_lora import HomoLoRAModel
from flcore.clients.clienthomolora import clientHomoLoRA
import concurrent.futures
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data
import time
import os
import h5py


class serverHomoLoRA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.rs_auc = []
        self.rs_test_acc = []
        self.rs_time_cost = []
        self.updates = []
        self.result_path = "../results/GLUE/"

        # Initialize test loader
        self.test_loader = self.load_test_data()

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientHomoLoRA)

        # Define loss function
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
                    accuracy, _ = self.evaluate(
                        self.global_model, self.test_loader)
                    self.rs_test_acc.append(accuracy)

                future_to_client = {executor.submit(
                    client.train, client.model): client for client in self.selected_clients}
                concurrent.futures.wait(future_to_client)

                self.receive_models()
                self.aggregate_parameters()

                end_time = time.time()
                self.rs_time_cost.append(end_time - start_time)

        print("\nBest accuracy.")
        if self.rs_test_acc:
            print(max(self.rs_test_acc))
        else:
            print("No accuracy results found.")

        print("\nAverage time cost per round.")
        if len(self.rs_time_cost) > 1:
            print(sum(self.rs_time_cost[1:]) / len(self.rs_time_cost[1:]))
        else:
            print("No time cost data available.")

        self.save_results()
        self.save_global_model()

    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.model = HomoLoRAModel(
                self.global_model, self.args.lora_rank, self.args.lora_alpha, self.args.lora_dropout)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        self.updates = []
        for client in self.selected_clients:
            client_model = client.model
            self.updates.append(client_model.state_dict())

    def aggregate_parameters(self):
        assert (len(self.updates) > 0)
        agg_state_dict = self.global_model.state_dict()

        for key in agg_state_dict:
            if key.startswith('lora_layers.'):
                agg_state_dict[key] = torch.mean(
                    torch.stack([update[key] for update in self.updates]), dim=0)

        self.global_model.load_state_dict(agg_state_dict)

        # 在服务器端对聚合后的全局模型进行微调
        server_optimizer = torch.optim.Adam(
            self.global_model.parameters(), lr=1e-4)
        server_criterion = torch.nn.CrossEntropyLoss()

        # 准备服务器端的训练数据
        server_trainloader = self.load_server_data()

        self.global_model.train()
        for _ in range(10):  # 微调轮次数
            for batch in server_trainloader:
                server_optimizer.zero_grad()

                # 前向传递和损失计算
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(
                        self.device, dtype=torch.long)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(
                            self.device, dtype=torch.long)
                    labels = batch['labels'].to(self.device, dtype=torch.long)
                else:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device, dtype=torch.long)
                    attention_mask = attention_mask.to(
                        self.device, dtype=torch.long)
                    labels = labels.to(self.device, dtype=torch.long)

                outputs = self.global_model(input_ids, attention_mask)
                loss = server_criterion(outputs['logits'], labels)

                loss.backward()
                server_optimizer.step()

        self.updates = []

    def evaluate(self, model, testloader):
        model.eval()
        correct = 0
        total = 0
        test_acc = 0
        test_num = 0
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
                labels = labels.to(self.device, dtype=torch.long)

                if attention_mask is not None:
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask)
                else:
                    outputs = model(input_ids=input_ids)

                test_acc += (torch.sum(torch.argmax(
                    outputs["logits"], dim=1) == labels)).item()

                test_num += labels.shape[0]

                y_prob.append(outputs["logits"].detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(
                    labels.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

                logits = outputs["logits"]
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

    def load_test_data(self):
        # Assuming the test data is stored with client id 0
        test_data = read_client_data(self.dataset, 0, is_train=False)
        return DataLoader(test_data, batch_size=self.args.batch_size, drop_last=False, shuffle=False)

    def load_server_data(self):
        # 加载服务器端的训练数据
        server_data = read_client_data(
            self.dataset, 0, is_train=True)  # 假设服务器端的数据客户端ID为-1
        server_trainloader = DataLoader(
            server_data, batch_size=self.args.batch_size, shuffle=True)
        return server_trainloader

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
