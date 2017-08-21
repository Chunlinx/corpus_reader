# -*- coding: utf-8 -*-

import cPickle as pkl
import os

class CBT(object):
    """
    Children's Book Test corpus
    """

    def __init__(self, path, word_type):
        assert word_type in ["NE", "CN", "V", "P"]
        self.word_type = word_type

        filenames = os.listdir(os.path.join(path, "data"))
        path_train = os.path.join(path, "data", 
            [n for n in filenames if n.startswith("cbtest_%s_train" % word_type)][0])
        path_val = os.path.join(path, "data",
            [n for n in filenames if n.startswith("cbtest_%s_valid_" % word_type)][0])
        path_test = os.path.join(path, "data",
            [n for n in filenames if n.startswith("cbtest_%s_test_" % word_type)][0])

        data_train = self.load_split(path_train)
        data_val = self.load_split(path_val)
        data_test = self.load_split(path_test)
        self.data = {
                "train": data_train,
                "val": data_val,
                "test": data_test}

    def load_split(self, path):
        data = []
        example = {"context": [],
                    "question": None,
                    "answer": None,
                    "candidates": None}
        for line in open(path):
            # skip empty lines
            line = line.decode("utf-8").strip()
            if line == "":
                continue
            # extract a sentence ID
            sent_id = int(line[0:line.find(" ")])
            content = line[line.find(" ")+1:]
            if sent_id == 21:
                # question & answer & candidates
                content = content.split("\t")
                content = [x for x in content if x != ""]
                assert len(content) == 3
                example["question"] = content[0].strip()
                example["answer"] = content[1].strip()
                example["candidates"] = content[2].split("|")
                assert example["answer"] in example["candidates"]
                data.append(example)
                # new example
                example = {"context": [],
                            "question": None,
                            "answer": None,
                            "candidates": None}
            else:
                # context
                example["context"].append(content.strip())
        return data

    def get(self, split):
        return self.data[split]

    def dump(self, path_dir):
        for split in ["train", "val", "test"]:
            data = self.get(split)
            pkl.dump(data, open(os.path.join(path_dir,
                "cbt.type_%s.%s.pkl" % (self.word_type, split)), "wb"))
