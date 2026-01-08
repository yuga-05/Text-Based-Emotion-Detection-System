import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

TOKENIZER_NAME = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_NAME)

MAX_LEN = 128

class EmotionDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["text"])
        label = int(self.data.iloc[idx]["label"])

        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }
train_dataset = EmotionDataset(os.path.join(DATA_DIR, "train.csv"))
valid_dataset = EmotionDataset(os.path.join(DATA_DIR, "valid.csv"))
test_dataset  = EmotionDataset(os.path.join(DATA_DIR, "test.csv"))

print("Tokenization completed")
print("Train samples:", len(train_dataset))
print("Valid samples:", len(valid_dataset))
print("Test samples :", len(test_dataset))
