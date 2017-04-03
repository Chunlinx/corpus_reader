# -*- coding: utf-8 -*-

import os

import numpy as np

label2id = {
    "non-paraphrase": 0,
    "paraphrase": 1}
id2label = {i:l for l,i in label2id.items()}

class MSRP(object):
    """
    Microsoft Research Paraphrase Corpus
    """
    def __init__(self, path):
        self.load_dataset(path)

    def load_dataset(self, path):
        def load_split(path, split):
            labels, sents_A, sents_B = [], [], []
            lines = open(os.path.join(path, "msr_paraphrase_%s.txt" % split)).readlines()
            lines = lines[1:]
            for line in lines:
                fields = line.strip().decode("utf-8").split("\t")
                assert len(fields) == 5
                label = int(fields[0])
                sent_A = fields[3]
                sent_B = fields[4]
                labels.append(label)
                sents_A.append(sent_A)
                sents_B.append(sent_B)
            labels = np.asarray(labels, dtype="int")
            return sents_A, sents_B, labels

        train_sents_A, train_sents_B, train_labels = load_split(path, "train")
        test_sents_A, test_sents_B, test_labels = load_split(path, "test")

        self.sentences = {
            "train": (train_sents_A, train_sents_B),
            "test": (test_sents_A, test_sents_B)}
        self.labels = {
            "train": train_labels,
            "test": test_labels}


    def get_data(self, split):
        return self.sentences[split][0], self.sentences[split][1], self.labels[split]

