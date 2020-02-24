import numpy as np
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

def preprocess_text(text):
    """Preprocess one sentence: tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a string of tokens joined by whitespace."""
    stemmer = PorterStemmer()
    stops = set(stopwords.words('english'))
    toks = word_tokenize(text)
    toks_stemmed = [stemmer.stem(tok.lower()) for tok in toks]
    toks_nopunc = [tok for tok in toks_stemmed if tok not in string.punctuation]
    toks_nostop = [tok for tok in toks_nopunc if tok not in stops]
    return " ".join(toks_nostop)

# TODO: lab, homework
def parse_sts(data_file):
    """
    Reads a tab-separated sts benchmark file and returns
    texts: list of tuples (text1, text2)
    labels: list of floats
    """
    texts = []
    labels = []

    with open(data_file, 'r') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(float(fields[4]))
            t1 = fields[5].lower()
            t2 = fields[6].lower()
            texts.append((t1, t2))
    return texts, labels

# TODO: lab, homework
def sts_to_pi(texts, sts_labels, max_nonparaphrase, min_paraphrase):
    """Convert a dataset from semantic textual similarity to paraphrase.
    Remove any examples that are > max_nonparaphrase and < min_paraphrase."""
    return texts, sts_labels
