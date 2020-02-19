import argparse
from util import parse_sts, sts_to_pi, preprocess_text
import numpy as np


def load_X(sent_pairs, tfidf_vectorizer):
    """Create a matrix where every row is a pair of sentences and every column in a feature.
    Feature (column) order is not important to the algorithm."""

    features = ["BLEU", "Word Error Rate", "Tfidf"]

    X = np.zeros((len(sent_pairs), len(features)))

    return X


def main(sts_train_file, sts_dev_file):
    """Fits a logistic regression for paraphrase identification, using string similarity metrics as features.
    Prints accuracy on held-out data. Data is formatted as in the STS benchmark"""

    min_paraphrase = 4.0
    max_nonparaphrase = 3.0

    # loading train
    train_texts_sts, train_y_sts = parse_sts(sts_train_file)

    # loading dev
    dev_texts_sts, dev_y_sts = parse_sts(sts_dev_file)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_dev_file", type=str, default="../strings_for_similarity/stsbenchmark/sts-dev.csv",
                        help="dev file")
    parser.add_argument("--sts_train_file", type=str, default="../strings_for_similarity/stsbenchmark/sts-train.csv",
                        help="train file")
    args = parser.parse_args()

    main(args.sts_train_file, args.sts_dev_file)
