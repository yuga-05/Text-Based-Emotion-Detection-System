import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from torch.optim import AdamW
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

print("Accelerator initialized")
print("Device:", device)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from tokenization import EmotionDataset
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "emotion_model")

os.makedirs(MODEL_DIR, exist_ok=True)

print("Data directory:", DATA_DIR)
train_csv = os.path.join(DATA_DIR, "train.csv")
valid_csv = os.path.join(DATA_DIR, "valid.csv")

if not os.path.exists(train_csv):
    raise FileNotFoundError(f"train.csv not found at {train_csv}")

train_dataset = EmotionDataset(train_csv)
valid_dataset = EmotionDataset(valid_csv)

print("Train samples:", len(train_dataset))
print("Valid samples:", len(valid_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=6
)
optimizer = AdamW(model.parameters(), lr=2e-5)
model, optimizer, train_loader = accelerator.prepare(
    model, optimizer, train_loader
)

print("Training started")

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs.loss
        total_loss += loss.item()

        accelerator.backward(loss)
        optimizer.step()

        if step % 50 == 0:
            accelerator.print(
                f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}"
            )

    avg_loss = total_loss / len(train_loader)
    accelerator.print(
        f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}"
    )

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    model.save_pretrained(MODEL_DIR)
    print("Model saved at:", MODEL_DIR)
