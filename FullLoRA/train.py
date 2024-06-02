import argparse
from models_lora import LoRAModel
import h5py
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import time
import torch
import pandas as pd
import os

class GLUEDataDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        sentence = row["sentence"]
        if not isinstance(sentence, str):
            raise ValueError(f"Expected string for 'sentence' but got {type(sentence)}")
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        label = torch.tensor(row["label"], dtype=torch.long)
        return inputs, label


def load_data(data_dir, tokenizer, max_len=128):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train_dataset = GLUEDataDataset(train_df, tokenizer, max_len)
    val_dataset = GLUEDataDataset(val_df, tokenizer, max_len)
    test_dataset = GLUEDataDataset(test_df, tokenizer, max_len)

    return train_dataset, val_dataset, test_dataset


def train(model, train_loader, optimizer, device, num_epochs=3):
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    loss_fn = torch.nn.CrossEntropyLoss()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            labels = labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions.double() / total_predictions
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

def save_results(rs_test_acc, rs_time_cost, file_path):
    with h5py.File(file_path, 'w') as hf:
        hf.create_dataset('rs_test_acc', data=rs_test_acc)
        hf.create_dataset('rs_time_cost', data=rs_time_cost)

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    base_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2)

    model = LoRAModel(base_model, args.lora_rank, args.lora_alpha, args.lora_dropout).to(device)

    print("Model Information:")
    print(model)

    train_dataset, val_dataset, test_dataset = load_data(args.data_dir, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Training Parameters:\n  Number of Epochs: {args.num_epochs}\n  Batch Size: {args.batch_size}\n  Learning Rate: {args.learning_rate}")

    rs_test_acc = []
    rs_time_cost = []

    for epoch in range(args.num_epochs):
        start_time = time.time()
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        train(model, train_loader, optimizer, device, num_epochs=1)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        rs_test_acc.append(val_accuracy)
        end_time = time.time()
        rs_time_cost.append(end_time - start_time)

    # Evaluate the model on the test set and print the results
    test_loss, test_accuracy = evaluate(model, test_loader, device)

    result_path = "../results/GLUE/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_path = os.path.join(result_path, "CoLA_FULLLoRA_results.h5")
    save_results(rs_test_acc, rs_time_cost, file_path)
    print(f"Results saved to {file_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a LoRA model on the GLUE dataset"
    )
    parser.add_argument(
        "--data_dir",
       type=str,
       default="GLUE/CoLA_full",
       help="Path to the dataset directory",
    )
    parser.add_argument(
        "--batch_size",
       type=int,
       default=32,
       help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
       type=float,
       default=5e-5,
       help="Learning rate for the optimizer",
    )
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="LoRA dropout rate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
