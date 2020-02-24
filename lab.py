import argparse
import numpy as np
from util import parse_sts, preprocess_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_score, recall_score

def main(sts_data):
    """Transform a semantic textual similarity dataset into a paraphrase identification.
    Data is formatted as in the STS benchmark"""

    max_nonparaphrase = 3.0
    min_paraphrase = 4.0

    # read the dataset
    texts, labels = parse_sts(sts_data)
    labels = np.asarray(labels)

    # get an array of the rows where the labels are in the right interval
    # I like this numpy
    #pi_rows = np.where(np.logical_or(labels>=min_paraphrase, labels<=max_nonparaphrase))[0]
    # here's a loop if you don't like numpy
    pi_rows = [i for i,label in enumerate(labels) if label >=min_paraphrase or label<=max_nonparaphrase]
    print(pi_rows)
    pi_texts = [texts[i] for i in pi_rows]
    # 1221 for dev
    print(f"{len(pi_texts)} sentence pairs kept")

    # using indexing to get the right rows out of labels
    pi_y = labels[pi_rows]
    # convert to binary using threshold
    pi_y = pi_y > max_nonparaphrase
    print(pi_y)

    # check your split agrees with mine
    num_nonparaphrase = (pi_y == False).sum()
    num_paraphrase = (pi_y == True).sum()
    # 957 for dev
    print(f"{num_nonparaphrase} non-paraphrase")
    # 264 for dev
    print(f"{num_paraphrase} paraphrase")


    # Instantiate a TFIDFVectorizer to create representations for sentences
    # get a single list of texts to determine vocabulary and document frequency
    all_t1, all_t2 = zip(*texts)
    all_texts = all_t1 + all_t2
    preproc_train_texts = [preprocess_text(text) for text in all_texts]

    preproc_vectorizer = TfidfVectorizer("content", lowercase=True, analyzer="word",
                                         token_pattern="\S+", use_idf=True, min_df=10)
    preproc_vectorizer.fit(preproc_train_texts)

    # compute cosine similarity for each pair of sentences
    # use a threshold of 0.7 to convert each similarity score into a paraphrase prediction
    cos_sims_preproc = []

    for t1,t2 in pi_texts:

        t1_preproc = preprocess_text(t1)
        t2_preproc = preprocess_text(t2)
        pair_reprs = preproc_vectorizer.transform([t1_preproc, t2_preproc])
        pair_similarity = cosine_similarity(pair_reprs[0], pair_reprs[1])
        cos_sims_preproc.append(pair_similarity[0,0])

    predictions = np.asarray(cos_sims_preproc) > 0.7


    # calculate and print precision and recall statistics for your system
    num_pred = predictions.sum()
    print(f"Number predicted paraphrase: {num_pred}")

    num_pos = pi_y.sum()
    print(f"Number positive: {num_pos}")

    true_pos = ((pi_y*1 + predictions*1) == 2)
    num_true_pos = true_pos.sum()
    print(f"Number true positive: {num_true_pos}")

    precision = num_true_pos / num_pred
    recall = num_true_pos / num_pos
    print(f"Scores: precision {precision:0.03}\trecall {recall:0.03}")

    # double check our work: use sklearn's implementation
    p = precision_score(pi_y, predictions)
    r = recall_score(pi_y, predictions)
    f = f1_score(pi_y, predictions)
    print(f"Scores: precision {p:0.03}\trecall {r:0.03}\tf1 {f:0.03}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="../strings_for_similarity/stsbenchmark/sts-dev.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)
