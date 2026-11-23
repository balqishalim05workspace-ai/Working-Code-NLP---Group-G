
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Locate project folder
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 2. Load preprocessed dataset
DATA_PATH = PROJECT_DIR / "data" / "preprocessed_dataset.csv"

df = pd.read_csv(DATA_PATH, encoding="utf-8")

print("Dataset loaded successfully!")
print("Total records:", len(df))


# 3. Check whether human annotations are available
if "human_label" not in df.columns:
    print("\nHuman annotations are not available yet.")
    print("Baseline model is prepared but cannot be trained.")
    print("Complete human annotation before training.")
    raise SystemExit


# 4. Prepare input and labels
df = df.dropna(subset=["human_label", "model_text"])

X = df["model_text"]
y = df["human_label"]

# 5. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 6. Create Logistic Regression pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=2,
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# 7. Train model
model.fit(X_train, y_train)

# 8. Generate predictions
predictions = model.predict(X_test)

# 9. Evaluate model
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["Non-sarcastic", "Sarcastic"]
))

# 10. Save trained model
import joblib

OUTPUT_PATH = PROJECT_DIR / "data" / "baseline_model.pkl"
joblib.dump(model, OUTPUT_PATH)

print("\nBaseline model trained successfully!")
print("Model saved to:", OUTPUT_PATH)
