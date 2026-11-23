

import pandas as pd
import re
import unicodedata
from pathlib import Path

# 1. Locate project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Load your cleaned dataset
DATA_PATH = PROJECT_DIR / "dataset_nlp_FINAL.csv"

df = pd.read_csv(DATA_PATH, encoding="cp1252")

print("Dataset loaded successfully!")
print("Original records:", len(df))


# 3. Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove unnecessary extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# 4. Prepare dataset
def prepare_dataset(df):
    df = df.copy()

    # Process original post and reply text
    df["original_post_text"] = (
        df["original_post_text"].apply(preprocess_text)
    )

    df["text_content"] = (
        df["text_content"].apply(preprocess_text)
    )

    # Combine original post context and reply
    df["model_text"] = (
        "Context: " + df["original_post_text"] +
        " Reply: " + df["text_content"]
    )

    return df


# 5. Apply preprocessing
df = prepare_dataset(df)


# 6. Save preprocessed dataset
OUTPUT_PATH = PROJECT_DIR / "data" / "preprocessed_dataset.csv"

df.to_csv(OUTPUT_PATH, index=False)

print("\nPreprocessing completed successfully!")
print("Total records:", len(df))
print("Saved file:", OUTPUT_PATH)

# 7. Display sample results
print("\nSample preprocessed data:")
print(
    df[["original_post_text", "text_content", "model_text"]].head()
)

