import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------------------------
# EARLY PRINT (prevents blank terminal)
# -------------------------------------------------
print("🚀 evaluate.py started", flush=True)

# -------------------------------------------------
# Fix Python path
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

from tokenization import EmotionDataset

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "emotion_model")

print("📂 Data directory:", DATA_DIR, flush=True)
print("📦 Model directory:", MODEL_DIR, flush=True)

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🖥️ Device:", device, flush=True)

# -------------------------------------------------
# Load tokenizer & model
# -------------------------------------------------
print("⏳ Loading tokenizer...", flush=True)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("⏳ Loading trained model...", flush=True)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

print("✅ Model loaded successfully", flush=True)

# -------------------------------------------------
# Load test dataset
# -------------------------------------------------
test_csv = os.path.join(DATA_DIR, "test.csv")

if not os.path.exists(test_csv):
    raise FileNotFoundError(f"❌ test.csv not found at {test_csv}")

test_dataset = EmotionDataset(test_csv)
test_loader = DataLoader(test_dataset, batch_size=8)

print("📊 Test samples:", len(test_dataset), flush=True)

# -------------------------------------------------
# Evaluation loop
# -------------------------------------------------
print("🔍 Starting evaluation loop...", flush=True)

all_preds = []
all_labels = []

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # progress indicator every 100 batches
        if step % 100 == 0:
            print(f"  ▶ Processed batch {step}", flush=True)

print("✅ Evaluation loop finished", flush=True)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
emotion_names = [
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "disgust"
]

print("\n📊 Classification Report:\n", flush=True)
print(classification_report(all_labels, all_preds, target_names=emotion_names))

print("🧩 Confusion Matrix:\n", flush=True)
print(confusion_matrix(all_labels, all_preds))

print("\n🎉 Evaluation completed successfully", flush=True)
