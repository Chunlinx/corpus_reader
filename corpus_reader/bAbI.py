# -*- coding: utf-8 -*-

import cPickle as pkl
import copy
import os

class BABI(object):

    def __init__(self, path, task_id):
        assert task_id in range(1, 21)
        self.task_id = task_id

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
        supports_map = {}
        for line in open(path):
            line = line.decode("utf-8")
            sent_id = int(line[0:line.find(" ")])
            if sent_id == 1:
                # new episode
                episode = {"context": [],
                            "question": None,
                            "answer": None,
                            "supports": None}
                supports_map = {}
            line = line.strip().replace(".", " . ").replace("?", " ? ")
            content = line[line.find(" ")+1:]
            if content.find("?") == -1:
                # context
                episode["context"].append(content.strip())
                supports_map[sent_id] = len(supports_map)
            else:
                # question & answers
                content = content.split("\t")
                assert len(content) == 3
                episode["question"] = content[0].strip()
                episode["answer"] = content[1].strip()
                episode["supports"] = [int(x) for x in content[2].split()]
                episode["supports"] = [supports_map[x] for x in episode["supports"]]
                episodes.append(copy.deepcopy(episode))
        return episodes

    def get(self, split):
        return self.episodes[split]

    def dump(self, path_dir):
        for split in ["train", "test"]:
            episodes = self.get(split)
            pkl.dump(episodes, open(os.path.join(path_dir,
                "babi.task_%02d.%s.pkl" % (self.task_id, split)), "wb"))

