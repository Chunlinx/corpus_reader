# -*- coding: utf-8 -*-

import copy
import os

class BABI(object):

    def __init__(self, path, task_id):
        self.load_dataset(path, task_id)

    def load_dataset(self, path, task_id):
        filenames = os.listdir(path)
        filenames = [n for n in filenames if n.startswith("qa%d_" % task_id)]
        assert len(filenames) == 2
        path_train = os.path.join(path, [n for n in filenames if "train" in n][0])
        path_test = os.path.join(path, [n for n in filenames if "test" in n][0])
        episodes_train = self.load_split(path_train)
        episodes_test = self.load_split(path_test)
        self.episodes = {
                "train": episodes_train,
                "test": episodes_test}

    def load_split(self, path):
        """
        Load a {train,test} corpus
        """
        episodes = []
        episode = None
        for i, line in enumerate(open(path)):
            sent_id = int(line[0:line.find(" ")])
            if sent_id == 1:
                # new episode
                episode = {"context": [],
                            "question": None,
                            "answer": None}
            line = line.strip().replace(".", " . ").replace("?", " ? ")
            content = line[line.find(" ")+1:]
            if content.find("?") == -1:
                # context
                episode["context"].append(content.strip())
            else:
                # question & answers
                content = content.split("\t")
                assert len(content) == 3
                episode["question"] = content[0].strip()
                episode["answer"] = content[1].strip()
                episode["supports"] = content[2].split()
                episodes.append(copy.deepcopy(episode))
        return episodes

    def get_data(self, split):
        return self.episodes[split]
