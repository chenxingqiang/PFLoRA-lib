import numpy as np
import time

import torch
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from flcore.trainmodel.hetlora_model import HetLoRAModel


class clientHetLoRA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.hetlora_min_rank = args.hetlora_min_rank
        self.hetlora_max_rank = args.hetlora_max_rank
        self.hetlora_gamma = args.hetlora_gamma
        self.local_learning_rate = args.local_learning_rate

        train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.HetLoRA = HetLoRAModel(
            self.model, self.hetlora_min_rank, self.hetlora_max_rank, self.hetlora_gamma)

    def train(self, net):
        trainloader = self.load_train_data()
        self.model.load_state_dict(net.state_dict())
        self.model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        print(f"Client {self.id} starts training:")
        print(f"  Number of training samples: {self.train_samples}")
        print(f"  Local epochs: {max_local_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.local_learning_rate}")

        for epoch in range(max_local_epochs):
            epoch_loss = 0.0
            for i, batch in enumerate(trainloader):
                if isinstance(batch, dict):
                    input_ids = batch['input_ids'].to(
                        self.device, dtype=torch.long)
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(
                            self.device, dtype=torch.long)
                    labels = batch['labels'].to(self.device, dtype=torch.long)
                else:
                    if len(batch) == 3:
                        input_ids, attention_mask, labels = batch
                        input_ids = input_ids.to(self.device, dtype=torch.long)
                        attention_mask = attention_mask.to(
                            self.device, dtype=torch.long)
                        labels = labels.to(self.device, dtype=torch.long)
                    elif len(batch) == 2:
                        input_ids, labels = batch
                        input_ids = input_ids.to(self.device, dtype=torch.long)
                        labels = labels.to(self.device, dtype=torch.long)
                        attention_mask = None
                    else:
                        raise ValueError("Unexpected batch format")

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                if attention_mask is not None:
                    output = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask)
                else:
                    output = self.model(input_ids=input_ids)

                loss = self.loss(output, labels)
                epoch_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            epoch_loss /= len(trainloader)
            print(
                f"  Epoch {epoch+1}/{max_local_epochs} | Loss: {epoch_loss:.4f}")

        self.HetLoRA.rank_self_pruning()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        print(f"Client {self.id} finished training.")

        # Add LoRA layers' state dict to the client model's state dict
        client_state_dict = {}
        client_state_dict = self.model.state_dict()
        client_state_dict['lora_layers'] = self.HetLoRA.lora_layers.state_dict()
        return self.HetLoRA.state_dict()

    def local_initialization(self, received_global_model):
        self.HetLoRA.truncate_lora_layer(received_global_model)
