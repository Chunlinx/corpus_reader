# -*- coding: utf-8 -*-

import json
import os

import numpy as np

import utils

label2id = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2}
id2label = {i:l for l,i in label2id.items()}

class SNLI(object):
    """
    Stanford Natural Language Inference (SNLI) corpus
    """

    def __init__(self, path):
        def process_binary_parse(parse):
            # words = parse.split()
            # words = [w for w in words if not w in ["(", ")"]]
            # sent = " ".join(words)
            # return sent 
            words = parse.replace("(", " ").replace(")", " ").replace("-LRB-", "(").replace("-RRB-", ")").split()
            return " ".join(words)

        def load_split(path, filename):
            labels, sents_A, sents_B = [], [], []
            image_names = []
            for line in open(os.path.join(path, filename)).readlines():
                line = line.strip()
                data = json.loads(line)
                label = data["gold_label"]
                if label == "-":
                    continue
                sent_A = process_binary_parse(data["sentence1_binary_parse"])
                sent_B = process_binary_parse(data["sentence2_binary_parse"])
                label = label2id[label]
                sents_A.append(sent_A)
                sents_B.append(sent_B)
                labels.append(label)
                image_name = data["captionID"]
                index = image_name.rfind("jpg")
                if index != -1:
                    image_name = image_name[:index+3]
                image_names.append(image_name) # XXX
            labels = np.asarray(labels, dtype="int")
            return sents_A, sents_B, labels, image_names

        train_sents_A, train_sents_B, train_labels, train_image_names = load_split(path, "snli_1.0_train.jsonl")
        val_sents_A, val_sents_B, val_labels, val_image_names = load_split(path, "snli_1.0_dev.jsonl")
        test_sents_A, test_sents_B, test_labels, test_image_names = load_split(path, "snli_1.0_test.jsonl")

        self.sentences = {
                "train": (train_sents_A, train_sents_B),
                "val": (val_sents_A, val_sents_B),
                "test": (test_sents_A, test_sents_B)}
        self.labels = {
                "train": train_labels,
                "val": val_labels,
                "test": test_labels}
        self.image_names = {
                "train": train_image_names,
                "val": val_image_names,
                "test": test_image_names}

    def get(self, split):
        return self.sentences[split][0], self.sentences[split][1], self.labels[split], self.image_names[split]

    def dump(self, path_dir):
        for split in ["train", "val", "test"]:
            sents_p, sents_h, labels, image_names = self.get(split)
            labels = [str(l) for l in labels]
            utils.write_lines(sents_p,
                os.path.join(path_dir, "snli.%s.premise.txt" % split))
            utils.write_lines(sents_h,
                os.path.join(path_dir, "snli.%s.hypothesis.txt" % split))
            utils.write_lines(labels,
                os.path.join(path_dir, "snli.%s.labels.txt" % split))
            utils.write_lines(image_names,
                os.path.join(path_dir, "snli.%s.image_names.txt" % split))

