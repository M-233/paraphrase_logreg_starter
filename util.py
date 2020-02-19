
def preprocess_text(text):
    """Preprocess one sentence: tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a string of tokens joined by whitespace."""

    return text

# TODO: lab, homework
def parse_sts(data_file):
    """
    Reads a tab-separated sts benchmark file and returns
    texts: list of tuples (text1, text2)
    labels: list of floats
    """
    texts = []
    labels = []

    return texts, labels

# TODO: lab, homework
def sts_to_pi(texts, sts_labels, max_nonparaphrase, min_paraphrase):
    """Convert a dataset from semantic textual similarity to paraphrase.
    Remove any examples that are > max_nonparaphrase and < min_nonparaphrase."""

    return texts, sts_labels
