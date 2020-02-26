Paraphrase Identification using string similarity
---------------------------------------------------

This project examines string similarity metrics for paraphrase identification.
It converts semantic textual similarity data to paraphrase identification data using threshholds.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting paraphrase.


Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).


## Homework: pi_logreg.py

* Train a logistic regression for PI on the training data using three features:
    - BLEU
    - Word Error Rate
    - Cosine Similarity of TFIDF vectors
* Use the logistic regression implementation in `sklearn`.
* Update the readme:
    * 1-sentence description of the dataset
    * Report your model's accuracy, precision, recall and f1-measure on the dev set.
    * Comment on what these metrics tell you about your model's performance and 
compare to the baseline in lab.py (~5 sentences)

`python pi_logreg.py --sts_dev_file stsbenchmark/sts-dev.csv --sts_train_file stsbenchmark/sts-train.csv`

## lab.py

`lab.py` converts a STS dataset to PI and reports the number of remaining examples
and the distribution of paraphrase/nonparaphrase.
Then, it evaluates TFIDF vector similarity with a threshold of 0.7 as a model of paraphrase
using precision and recall.

Example usage:

`python lab.py --sts_data stsbenchmark/sts-dev.csv`


## Results

### Dataset descrtption

### Metrics result


               BLEU  Word Error Rate     Tfidf
accuracy   0.811630         0.805897  0.855856
precision  0.637097         0.589404  0.705607
recall     0.299242         0.337121  0.571970
f1         0.407216         0.428916  0.631799

baseline: accuracy 0.823   precision 0.586        recall 0.617    f1 0.601


