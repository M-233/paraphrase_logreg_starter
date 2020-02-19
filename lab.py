import argparse
import numpy as np
from util import parse_sts, preprocess_text, sts_to_pi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def main(sts_data):
    """Transform a semantic textual similarity dataset into a paraphrase identification.
    Data is formatted as in the STS benchmark"""

    max_nonparaphrase = 3.0
    min_paraphrase = 4.0

    # read the dataset
    texts, labels = parse_sts(sts_data)
    labels = np.asarray(labels)

    pi_texts, pi_labels = sts_to_pi(texts, labels)

    # calculate to check your split agrees with mine
    num_nonparaphrase = 0
    num_paraphrase = 0
    # 957 for dev
    print(f"{num_nonparaphrase} non-paraphrase")
    # 264 for dev
    print(f"{num_paraphrase} paraphrase")


    # Instantiate a TFIDFVectorizer to create representations for sentences
    # compute cosine similarity for each pair of sentences
    # use a threshold of 0.7 to convert each similarity score into a paraphrase prediction
    cos_sims_preproc = []

    predictions = np.asarray(cos_sims_preproc) > 0.7


    # calculate and print precision and recall statistics for your system
    num_pred = 0
    print(f"Number predicted paraphrase: {num_pred}")

    num_pos = 0
    print(f"Number positive: {num_pos}")

    num_true_pos = 0
    print(f"Number true positive: {num_true_pos}")

    precision = 0
    recall = 0
    print(f"Scores: precision {precision:0.03}\trecall {recall:0.03}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="../strings_for_similarity/stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)
