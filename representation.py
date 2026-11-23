
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

# 1. Locate project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Load preprocessed dataset
DATA_PATH = PROJECT_DIR / "data" / "preprocessed_dataset.csv"

df = pd.read_csv(DATA_PATH, encoding="utf-8")

print("Preprocessed dataset loaded successfully!")
print("Total records:", len(df))


# 3. Check that model_text column exists
if "model_text" not in df.columns:
    raise ValueError("The model_text column was not found!")

# 4. Prepare text data
texts = df["model_text"].fillna("").astype(str)

# 5. Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    sublinear_tf=True
)

# 6. Convert text into TF-IDF features
tfidf_features = vectorizer.fit_transform(texts)

# 7. Display feature representation information
print("\nTF-IDF Feature Representation Completed!")
print("Number of records:", tfidf_features.shape[0])
print("Number of TF-IDF features:", tfidf_features.shape[1])
print("Feature matrix dimensions:", tfidf_features.shape)

# 8. Save TF-IDF feature matrix
OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

save_npz(
    OUTPUT_DIR / "tfidf_features.npz",
    tfidf_features
)

# 9. Save feature names
feature_names = vectorizer.get_feature_names_out()

pd.DataFrame({
    "feature": feature_names
}).to_csv(
    OUTPUT_DIR / "tfidf_feature_names.csv",
    index=False
)

# 10. Save the fitted vectorizer
import joblib

joblib.dump(
    vectorizer,
    OUTPUT_DIR / "tfidf_vectorizer.pkl"
)

print("\nFiles saved successfully:")
print("- tfidf_features.npz")
print("- tfidf_feature_names.csv")
print("- tfidf_vectorizer.pkl")

# 11. Display sample TF-IDF features
print("\nSample TF-IDF features:")
print(feature_names[:20])
