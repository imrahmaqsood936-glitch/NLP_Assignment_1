
from pathlib import Path
import pandas as pd
import re
import unicodedata
from nltk.stem import PorterStemmer

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "imdb_reviews.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "cleaned_reviews.csv"

stemmer = PorterStemmer()

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "am", "be", "been", "being", "and", "or", "but",
    "of", "to", "in", "on", "at", "for", "with",
    "this", "that", "these", "those", "it", "its",
    "he", "she", "they", "them", "we", "you", "i",
    "my", "your", "our", "their", "me", "him", "her",
    "as", "by", "from", "so", "if", "then", "than",
    "can", "could", "would", "should",
    "have", "has", "had", "do", "does", "did"
}

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "it's": "it is",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "i've": "i have",
    "you've": "you have",
    "they've": "they have",
    "we've": "we have",
    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "they'll": "they will",
    "we'll": "we will"
}

def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower().replace("’", "'")

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\b\d{1,2}\s*/\s*10\b", " ", text)

    for contraction, expanded in sorted(
        CONTRACTIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):
        text = re.sub(
            re.escape(contraction) + r"(?![a-z])",
            lambda match, replacement=expanded: replacement,
            text
        )

    tokens = re.findall(r"\b[a-z0-9]+\b", text)

    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in STOPWORDS
    ]

    return " ".join(cleaned_tokens)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "review" not in df.columns:
        raise ValueError("Dataset must contain a review column.")

    before_tokens = [
        word
        for review in df["review"].fillna("")
        for word in re.findall(
            r"\b[a-z0-9]+\b",
            unicodedata.normalize("NFKC", str(review)).lower()
        )
    ]

    df["cleaned_review"] = df["review"].fillna("").apply(clean_text)

    after_tokens = [
        word
        for review in df["cleaned_review"]
        for word in review.split()
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print("Cleaning completed successfully!")
    print("Total reviews:", len(df))
    print("Vocabulary before cleaning:", len(set(before_tokens)))
    print("Vocabulary after cleaning:", len(set(after_tokens)))

    print("\nOriginal review:")
    print(df["review"].iloc[0][:300])

    print("\nCleaned review:")
    print(df["cleaned_review"].iloc[0][:300])

    print("\nSaved at:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
