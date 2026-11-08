from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "IMDB Dataset.csv"
OUTPUT_FILE = PROCESSED_DIR / "imdb_reviews.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)
df.columns = df.columns.str.strip().str.lower()
df = df[["review", "sentiment"]].dropna()

df["review"] = df["review"].astype(str)
df["sentiment"] = df["sentiment"].str.strip().str.lower()

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

print("Dataset prepared successfully!")
print("Total reviews:", len(df))
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())
print("\nSaved at:", OUTPUT_FILE)
