import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "emotion_model")

emotion_map = {
    0: "Joy",
    1: "Sadness",
    2: "Anger",
    3: "Fear",
    4: "Surprise",
    5: "Disgust"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

TEMPERATURE = 1.7
TOP_K = 3

def predict_emotion(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits / TEMPERATURE, dim=1).squeeze()
    top_probs, top_indices = torch.topk(probs, TOP_K)

    return [
        {
            "emotion": emotion_map[i.item()],
            "score": round(p.item(), 2)
        }
        for p, i in zip(top_probs, top_indices)
    ]
