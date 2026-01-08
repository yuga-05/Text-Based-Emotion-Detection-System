import os
import pandas as pd
from datasets import load_dataset

BASE_DIR = "emotion_detection"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

dataset = load_dataset("go_emotions")

GO_EMOTION_MAP = {
    0: "admiration", 1: "amusement", 2: "anger", 3: "annoyance",
    4: "approval", 5: "caring", 6: "confusion", 7: "curiosity",
    8: "desire", 9: "disappointment", 10: "disapproval",
    11: "disgust", 12: "embarrassment", 13: "excitement",
    14: "fear", 15: "gratitude", 16: "grief", 17: "joy",
    18: "love", 19: "nervousness", 20: "optimism", 21: "pride",
    22: "realization", 23: "relief", 24: "remorse",
    25: "sadness", 26: "surprise", 27: "neutral"
}

TARGET_EMOTIONS = {
    "joy": 0,
    "sadness": 1,
    "anger": 2,
    "fear": 3,
    "surprise": 4,
    "disgust": 5
}

def filter_emotions(split):
    texts = []
    labels = []

    for item in split:
        for label_id in item["labels"]:
            emotion = GO_EMOTION_MAP[label_id]
            if emotion in TARGET_EMOTIONS:
                texts.append(item["text"])
                labels.append(TARGET_EMOTIONS[emotion])
                break 

    return pd.DataFrame({
        "text": texts,
        "label": labels
    })

train_df = filter_emotions(dataset["train"])
valid_df = filter_emotions(dataset["validation"])
test_df  = filter_emotions(dataset["test"])

train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
valid_df.to_csv(os.path.join(PROCESSED_DIR, "valid.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

print("Dataset collection & filtering completed")
print("Train size:", len(train_df))
print("Valid size:", len(valid_df))
print("Test size :", len(test_df))
