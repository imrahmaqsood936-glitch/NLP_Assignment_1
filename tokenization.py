
from pathlib import Path
import re
import unicodedata
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer

# Project paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_DIR / "data" / "processed" / "imdb_reviews.csv"

# NLTK tokenizer (does not require punkt download)
nltk_tokenizer = TreebankWordTokenizer()

# Regex tokenizer
REGEX_PATTERN = re.compile(
    r"https?://\S+|www\.\S+|<[^>]+>|"
    r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*"
    r"(?:-[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*)*"
)

def regex_tokenize(text):
    text = unicodedata.normalize("NFKC", str(text))
    return REGEX_PATTERN.findall(text)


def nltk_tokenize(text):
    text = unicodedata.normalize("NFKC", str(text))
    return nltk_tokenizer.tokenize(text)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "review" not in df.columns:
        raise ValueError("Dataset must contain a review column.")

    # Use first 1000 reviews for comparison
    sample = df["review"].fillna("").head(1000)

    regex_tokens = []
    nltk_tokens = []

    for review in sample:
        regex_tokens.extend(regex_tokenize(review))
        nltk_tokens.extend(nltk_tokenize(review))

    # Dataset-specific patterns
    sample_text = " ".join(sample.astype(str))

    html_count = len(re.findall(r"<br\s*/?>", sample_text, flags=re.I))
    rating_count = len(re.findall(r"\b\d+(?:\.\d+)?\s*/\s*10\b", sample_text))
    url_count = len(re.findall(r"https?://\S+|www\.\S+", sample_text))
    hyphen_count = len(re.findall(r"\b\w+(?:-\w+)+\b", sample_text))

    print("TOKENIZATION COMPARISON")
    print("----------------------")
    print("Documents analyzed:", len(sample))

    print("\nRegex Tokenizer:")
    print("Total tokens:", len(regex_tokens))
    print("Unique tokens:", len(set(regex_tokens)))
    print("Type-token ratio:", round(len(set(regex_tokens)) / max(len(regex_tokens), 1), 4))

    print("\nNLTK Treebank Tokenizer:")
    print("Total tokens:", len(nltk_tokens))
    print("Unique tokens:", len(set(nltk_tokens)))
    print("Type-token ratio:", round(len(set(nltk_tokens)) / max(len(nltk_tokens), 1), 4))

    print("\nDATASET-SPECIFIC PATTERNS")
    print("------------------------")
    print("HTML break tags:", html_count)
    print("Rating expressions (e.g. 8/10):", rating_count)
    print("URLs:", url_count)
    print("Hyphenated words:", hyphen_count)

    print("\nTOKENIZATION EXAMPLE")
    print("--------------------")
    example = sample.iloc[0]
    print("Original review:")
    print(str(example)[:300])

    print("\nRegex tokens:")
    print(regex_tokenize(example)[:40])

    print("\nNLTK tokens:")
    print(nltk_tokenize(example)[:40])


if __name__ == "__main__":
    main()
