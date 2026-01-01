import os
import sys
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

# -------------------------------------------------
# Fix path
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(CURRENT_DIR)
MODEL_DIR = os.path.join(BASE_DIR, "models", "emotion_model")

# -------------------------------------------------
# Emotion labels
# -------------------------------------------------
emotion_map = {
    0: "Joy",
    1: "Sadness",
    2: "Anger",
    3: "Fear",
    4: "Surprise",
    5: "Disgust"
}

# -------------------------------------------------
# Config
# -------------------------------------------------
TOP_K = 3
TEMPERATURE = 1.7   # keeps predictions realistic

# -------------------------------------------------
# Device
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load model & tokenizer
# -------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

print("✅ Emotion detection ready (type 'exit' to quit)")

# -------------------------------------------------
# Inference loop
# -------------------------------------------------
while True:
    text = input("\n📝 Text: ")

    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Temperature-scaled softmax
    probs = F.softmax(outputs.logits / TEMPERATURE, dim=1).squeeze()

    # Get top 3 emotions
    top_probs, top_indices = torch.topk(probs, TOP_K)

    print("\n🎯 Detected Emotions:")
    for prob, idx in zip(top_probs, top_indices):
        emotion = emotion_map[idx.item()]
        print(f"{emotion}: {prob.item():.2f}")
