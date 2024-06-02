import random
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, r, lora_alpha, lora_dropout=0.1):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)
        self.lora_A = nn.Parameter(torch.zeros((input_dim, self.r)))
        self.lora_B = nn.Parameter(torch.zeros((self.r, output_dim)))
        self.scaling = self.lora_alpha / self.r

    def update_layer(self, base_layer):
        input_dim, output_dim = base_layer.weight.shape[1], base_layer.weight.shape[0]
        self.lora_A = nn.Parameter(torch.zeros((input_dim, self.r)))
        self.lora_B = nn.Parameter(torch.zeros((self.r, output_dim)))
        self.scaling = self.lora_alpha / self.r

    def forward(self, x):
        original_shape = x.shape
        x_flattened = x.view(-1, original_shape[-1])

        assert x_flattened.shape[1] == self.lora_A.shape[
            0], f"Shape mismatch: x_flattened.shape={x_flattened.shape}, lora_A.shape={self.lora_A.shape}"
        assert self.lora_A.shape[1] == self.lora_B.shape[
            0], f"Shape mismatch: lora_A.shape={self.lora_A.shape}, lora_B.shape={self.lora_B.shape}"

        lora_output = self.lora_dropout(
            x_flattened @ self.lora_A) @ self.lora_B * self.scaling
        lora_output = lora_output.view(*original_shape[:-1], -1)

        assert lora_output.shape == x.shape, f"Shape mismatch: lora_output.shape={lora_output.shape}, x.shape={x.shape}"
        output = x + lora_output

        return output


class HetLoRAModel(nn.Module):
    def __init__(self, base_model, hetlora_min_rank, hetlora_max_rank, hetlora_gamma, lora_alpha=1.0, lora_dropout=0.1):
        super(HetLoRAModel, self).__init__()
        self.base_model = base_model
        self.hetlora_min_rank = hetlora_min_rank
        self.hetlora_max_rank = hetlora_max_rank
        self.hetlora_gamma = hetlora_gamma
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_layers = self.add_lora_layers()
        self.original_norms = self.calculate_original_norms()

    def add_lora_layers(self):
        lora_layers = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                input_dim = module.weight.shape[1]
                output_dim = module.weight.shape[0]
                if input_dim != 768 or output_dim != 768:
                    continue
                lora_layer = LoRALayer(
                    input_dim, output_dim, self.hetlora_max_rank, self.lora_alpha, self.lora_dropout)
                lora_layers.append(lora_layer)
        return nn.ModuleList(lora_layers)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model.bert(
            input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        lora_index = 0
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] == 768 and module.weight.shape[1] == 768:
                sequence_output = self.lora_layers[lora_index](sequence_output)
                lora_index += 1

        pooled_output = self.base_model.bert.pooler(sequence_output)
        logits = self.base_model.classifier(pooled_output)
        return logits

    def calculate_original_norms(self):
        original_norms = []
        for lora_layer in self.lora_layers:
            original_norm = torch.norm(
                lora_layer.lora_A) * torch.norm(lora_layer.lora_B)
            original_norms.append(original_norm)
        return original_norms

    def truncate_lora_layer(self, global_model):
        global_lora_layers = global_model.lora_layers
        client_lora_rank = random.randint(
            self.hetlora_min_rank, self.hetlora_max_rank)
        for idx, global_layer in enumerate(global_lora_layers):
            self.lora_layers[idx].lora_A = nn.Parameter(
                global_layer.lora_A[:, :client_lora_rank])
            self.lora_layers[idx].lora_B = nn.Parameter(
                global_layer.lora_B[:client_lora_rank, :])
            self.lora_layers[idx].r = client_lora_rank

    def rank_self_pruning(self):
        for idx, lora_layer in enumerate(self.lora_layers):
            original_norm = self.original_norms[idx]
            ranks_to_prune = int(lora_layer.r * (1 - self.hetlora_gamma))
            pruned_norm = torch.norm(
                lora_layer.lora_A[-ranks_to_prune:, :]) * torch.norm(lora_layer.lora_B[:, -ranks_to_prune:])
            if pruned_norm < original_norm:
                lora_layer.lora_A = nn.Parameter(
                    lora_layer.lora_A[:-ranks_to_prune, :])
                lora_layer.lora_B = nn.Parameter(
                    lora_layer.lora_B[:, :-ranks_to_prune])
                lora_layer.r -= ranks_to_prune

    @staticmethod
    def zero_padding(global_lora_layer, client_lora_layer):
        pad_rows = global_lora_layer.lora_A.shape[0] - \
            client_lora_layer.lora_A.shape[0]
        pad_cols = global_lora_layer.lora_B.shape[1] - \
            client_lora_layer.lora_B.shape[1]
        client_lora_layer.lora_A = nn.functional.pad(
            client_lora_layer.lora_A, (0, 0, 0, pad_rows))
        client_lora_layer.lora_B = nn.functional.pad(
            client_lora_layer.lora_B, (0, pad_cols, 0, 0))
        return client_lora_layer

    @staticmethod
    def sparsity_weighted_aggregation(global_lora_layer, client_lora_layers):
        client_lora_layers = [HetLoRAModel.zero_padding(
            global_lora_layer, layer) for layer in client_lora_layers]
        sparsity_weights = []
        for client_layer in client_lora_layers:
            client_lora_matrix = client_layer.lora_A @ client_layer.lora_B
            sparsity_weights.append(torch.norm(client_lora_matrix, p='fro'))
        sparsity_weights = [w / sum(sparsity_weights)
                            for w in sparsity_weights]
        global_lora_layer.lora_A *= 0
        global_lora_layer.lora_B *= 0
        for w, client_layer in zip(sparsity_weights, client_lora_layers):
            global_lora_layer.lora_A += client_layer.lora_A * w
            global_lora_layer.lora_B += client_layer.lora_B * w
        return global_lora_layer
