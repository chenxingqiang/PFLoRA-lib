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

    def forward(self, x):
        original_shape = x.shape
        x_flattened = x.view(-1, original_shape[-1])
        lora_output = (
            self.lora_dropout(x_flattened @ self.lora_A) @ self.lora_B * self.scaling
        )
        lora_output = lora_output.view(*original_shape[:-1], -1)
        output = x + lora_output
        return output


class LoRAModel(nn.Module):
    def __init__(self, base_model, lora_rank, lora_alpha, lora_dropout):
        super(LoRAModel, self).__init__()
        self.base_model = base_model
        self.lora_layers = self.add_lora_layers(lora_rank, lora_alpha, lora_dropout)

    def add_lora_layers(self, lora_rank, lora_alpha, lora_dropout):
        lora_layers = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                input_dim = module.weight.shape[1]
                output_dim = module.weight.shape[0]
                if input_dim != 768 or output_dim != 768:
                    continue
                lora_layer = LoRALayer(
                    input_dim, output_dim, lora_rank, lora_alpha, lora_dropout
                )
                lora_layers.append(lora_layer)
        return nn.ModuleList(lora_layers)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model.bert(
            input_ids=input_ids, attention_mask=attention_mask
        )
        sequence_output = outputs.last_hidden_state

        for lora_layer in self.lora_layers:
            sequence_output = lora_layer(sequence_output)

        pooled_output = self.base_model.bert.pooler(sequence_output)
        logits = self.base_model.classifier(pooled_output)
        return {"logits": logits}
