# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd

label2id = {
    "ENTAILMENT": 0,
    "NEUTRAL": 1,
    "CONTRADICTION": 2}
id2label = {i:l for l,i in label2id.items()}

class SICK(object):
    """
    SICK corpus
    """

    def __init__(self, path):
        def load_split(filename):
            data = pd.read_csv(os.path.join(path, filename), sep="\t", header=0)
            sents_A = data["sentence_A"].values
            sents_B = data["sentence_B"].values
            labels = data["entailment_judgment"].values
            labels = np.asarray([label2id[l] for l in labels], dtype="int")
            scores = data["relatedness_score"].values
            return sents_A, sents_B, labels, scores

        train_sents_A, train_sents_B, train_labels, train_scores = load_split("SICK_train.txt")
        val_sents_A, val_sents_B, val_labels, val_scores = load_split("SICK_trial.txt")
        test_sents_A, test_sents_B, test_labels, test_scores = load_split("SICK_test_annotated.txt")

        self.sentences = {
                "train": (train_sents_A, train_sents_B),
                "val": (val_sents_A, val_sents_B),
                "test": (test_sents_A, test_sents_B)}
        self.labels = {
                "train": train_labels,
                "val": val_labels,
                "test": test_labels}
        self.scores = {
                "train": train_scores,
                "val": val_scores,
                "test": test_scores}

    def get(self, split):
        return self.sentences[split][0], self.sentences[split][1], self.labels[split], self.scores[split]
