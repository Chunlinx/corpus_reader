# -*- coding: utf-8 -*-

import os

import numpy as np
import pytreebank

class SST(object):
    """
    Stanford Sentiment Treebank (SST) corpus
    """
    
    def __init__(self, path):
        def load_split(path, split):
            data = pytreebank.import_tree_corpus(os.path.join(path, "%s.txt" % split))
            labels, sentences = [], []
            for example in data:
                max_length = -1
                main_label, main_sent = None, None
                for label, sent in example.to_labeled_lines():
                    if len(sent) > max_length:
                        max_length = len(sent)
                        main_label = label
                        main_sent = sent
                labels.append(main_label)
                sentences.append(main_sent)
            labels = np.asarray(labels, dtype="int")
            return sentences, labels

        train_sentences, train_labels = load_split(path, "train")
        val_sentences, val_labels = load_split(path, "dev")
        test_sentences, test_labels = load_split(path, "test")

        self.sentences = {
                "train": train_sentences,
                "val": val_sentences,
                "test": test_sentences}
        self.labels = {
                "train": train_labels,
                "val": val_labels,
                "test": test_labels}

    def get(self, split):
        return self.sentences[split], self.labels[split]
