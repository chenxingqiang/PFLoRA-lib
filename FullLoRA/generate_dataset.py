import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer

# 设置随机种子以保证可重复性
random_seed = 42
np.random.seed(random_seed)
batch_size = 10
train_ratio = 0.75

dir_path = "GLUE/CoLA"


def preprocess(tokenizer, examples):
    texts = (examples["sentence"],)
    result = tokenizer(*texts, padding="max_length", max_length=128, truncation=True)
    return result


def standardize_columns(file_path):
    df = pd.read_csv(file_path, delimiter="\t", header=None)
    expected_columns = {
        4: ["ignore", "label", "ignore2", "sentence"],
        3: ["label", "ignore", "sentence"],
        2: ["label", "sentence"],
    }
    if df.shape[1] in expected_columns:
        df.columns = expected_columns[df.shape[1]]
    else:
        raise ValueError(
            f"Unexpected number of columns ({df.shape[1]}) in the TSV file."
        )
    return df[["label", "sentence"]]


def load_glue_data(task_name):
    dataset_path = os.path.join(
        os.path.expanduser("~/datasets"), "glue", task_name.lower()
    )

    train_df = standardize_columns(os.path.join(dataset_path, "in_domain_train.tsv"))
    validation_df = standardize_columns(os.path.join(dataset_path, "in_domain_dev.tsv"))
    test_df = standardize_columns(os.path.join(dataset_path, "out_of_domain_dev.tsv"))

    datasets = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(validation_df),
            "test": Dataset.from_pandas(test_df),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    datasets = datasets.map(lambda x: preprocess(tokenizer, x), batched=True)

    return datasets


def split_and_save_data(datasets, output_dir, train_ratio):
    train_df = pd.DataFrame(datasets["train"])
    test_df = pd.DataFrame(datasets["test"])

    train_df, val_df = train_test_split(
        train_df, test_size=1 - train_ratio, random_state=random_seed
    )

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print(f"Data saved to {output_dir}")


def main():
    task_name = "CoLA"
    datasets = load_glue_data(task_name)

    output_dir = "GLUE/CoLA_full"
    split_and_save_data(datasets, output_dir, train_ratio)
    print("Dataset processing and saving completed.")


if __name__ == "__main__":
    main()
