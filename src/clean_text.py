import os
import re
import pandas as pd

# ----------------------------
# Correct project root
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_file(filename):
    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")

    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str).apply(clean_text)

    # Remove empty / very short text
    df = df[df["text"].str.len() > 3]

    # Remove duplicates
    df = df.drop_duplicates(subset="text")

    df.to_csv(path, index=False)
    print(f"✅ Cleaned {filename} | Rows: {len(df)}")

# ----------------------------
# Run cleaning
# ----------------------------
for file in ["train.csv", "valid.csv", "test.csv"]:
    clean_file(file)

print("🎉 Step 3 completed successfully")
