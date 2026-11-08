# Movie Reviews Sentiment Analysis Using NLP

## Assignment 1 – Dataset Exploration and Text Preprocessing

## Project Overview

This project explores and preprocesses the IMDB Movie Reviews Dataset using Natural Language Processing (NLP) techniques.

The dataset contains movie reviews labelled as positive or negative. The purpose of this assignment is to explore the dataset, compare tokenization methods, identify text patterns, and prepare cleaned reviews for sentiment analysis.

## Dataset Information

- **Dataset:** IMDB Movie Reviews Dataset
- **Source:** [Zenodo – IMDB Dataset](https://zenodo.org/records/7928582)
- **Total Reviews:** 50,000
- **Positive Reviews:** 25,000
- **Negative Reviews:** 25,000
- **Format:** CSV

**Note:** The original raw dataset is not included in this repository. Download it from the source and place the file named `IMDB Dataset.csv` in the project root directory, alongside `README.md`.

Please refer to the Zenodo record for the dataset's licence and usage conditions.

## Technologies and Libraries

- Python
- Pandas
- NLTK
- Matplotlib
- Regular Expressions (`re`)
- Unicode Normalization (`unicodedata`)

## Project Structure

```text
NLP_Assignment_1/
│
├── data/
│   └── processed/
│
├── src/
│   ├── download_dataset.py
│   ├── tokenization.py
│   └── clean_text.py
│
├── notebooks/
│   └── dataset_exploration.ipynb
│
├── reports/
│   ├── assignment1_report.md
│   └── assignment1_report.pdf
│
├── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/imrahhassan12-netizen/NLP_Assignment_1.git
cd NLP_Assignment_1
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare the Dataset

Download the IMDB dataset from:

[https://zenodo.org/records/7928582](https://zenodo.org/records/7928582)

Place the downloaded CSV file in the project root directory.

The expected filename is:

```text
IMDB Dataset.csv
```

### 6. Run Dataset Preparation

```bash
python src/download_dataset.py
```

### 7. Run Tokenization Comparison

```bash
python src/tokenization.py
```

### 8. Run Text Cleaning

```bash
python src/clean_text.py
```

The processed and cleaned datasets are saved in:

```text
data/processed/
```

## Dataset Exploration

The dataset contains 50,000 reviews with an equal distribution of positive and negative sentiments.

The exploration includes:

- Dataset dimensions and basic statistics.
- Review length analysis.
- Sentiment distribution.
- Vocabulary analysis.
- Frequent word analysis.
- Zipf's Law and long-tail distribution observations.
- Identification of messy text patterns.

The exploration notebook is available at:

`notebooks/dataset_exploration.ipynb`

## Tokenization

Two tokenization methods were compared:

1. Custom Regex Tokenizer.
2. NLTK Treebank Word Tokenizer.

The comparison includes total tokens, unique tokens, and Type-Token Ratio (TTR).

Dataset-specific patterns such as HTML tags, URLs, rating expressions, and hyphenated words were also examined.

## Text Preprocessing

The cleaning pipeline includes:

- Unicode normalization.
- HTML tag and noise removal.
- Lowercase conversion.
- Contraction handling.
- Tokenization.
- Stopword removal while preserving negation words.
- Porter stemming.

The cleaned dataset is saved as `cleaned_reviews.csv` in `data/processed/`.

## Team Members

- **Imrah** – Roll No: 0293
- **Ajwa** – Team Member

## Assignment Deliverables

- Dataset exploration notebook.
- Tokenization comparison.
- Regular expression analysis.
- Reusable text-cleaning pipeline.
- Cleaned dataset generation script.
- Assignment report in Markdown and PDF formats.

## Repository

GitHub: [NLP_Assignment_1](https://github.com/imrahhassan12-netizen/NLP_Assignment_1)