# -*- coding: utf-8 -*-

import bz2
from collections import OrderedDict
import os
import re

import bllipparser
import pyprind

class BllipWithoutParseTrees(object):
    """
    Brown Laboratory for Linguistic Information Processing (BLLIP)
    North American News Text Corpus, Complete
    Ignore parse trees
    """

    def __init__(self, path):
        self.path = path
        pass

    def dump(self, path_dir):
        # e.g., 50 reute9406_007.0356_13
        # 50 => # of parses
        # reute9406_007 =>  article id from the North Americal News Text Corpus
        # 13 => the index of the sentence in the article
        re_head = re.compile(r"([0-9]+) (.*)_([0-9]+)")
        re_tree = re.compile(r"^\(.*\)$")
        dir_list = os.listdir(self.path)
        dir_list.sort()
        for dir_name in pyprind.prog_bar(dir_list):
            if not os.path.exists(os.path.join(path_dir, dir_name)):
                os.makedirs(os.path.join(path_dir, dir_name))
            # Directories that contain error files: 271, 297, 299
            if int(dir_name.split(".")[0]) in [271, 297, 299]:
                print("Skipped.")
                continue
            file_names = os.listdir(os.path.join(self.path, dir_name))
            file_names.sort()
            for file_name in file_names:
                print("Processing %s ..." % os.path.join(self.path, dir_name, file_name))
                # Read 
                articles = OrderedDict()
                article_id_set = set()
                bz_file = bz2.BZ2File(os.path.join(self.path, dir_name, file_name))
                lines = bz_file.readlines()
                for line in lines:
                    # line = line.decode("latin8").strip()
                    line = line.strip()
                    if re_head.match(line):
                        # 新しい文への切り替え点
                        match = re_head.findall(line)
                        match = match[0]
                        # n_parses = int(match[0])
                        article_id = match[1]
                        sentence_index = int(match[2])
                        articles["%s/%d" % (article_id, sentence_index)] = None
                        article_id_set.add(article_id)
                    elif re_tree.match(line):
                        # 文. まだassignしてなければ読み込み，既にしてればスルー.
                        if articles["%s/%d" % (article_id, sentence_index)] is None:
                            # まだassignしてなければ読み込む
                            tree = bllipparser.Tree(line)
                            tokens = tree.tokens()
                            raw_sent = " ".join(tokens)
                            raw_sent = raw_sent.replace("-LRB-", "(").replace("-RRB-", ")")
                            articles["%s/%d" % (article_id, sentence_index)] = raw_sent.decode("latin8")
                        else:
                            # 既にassignしてれば何もしない (計算コストの節約)
                            continue
                    else:
                        # 恐らくparseスコア
                        continue
                # Wtite
                for article_id in article_id_set:
                    with open(os.path.join(path_dir, dir_name, article_id + ".txt"), "w") as f:
                        for key in articles:
                            key0, key1 = key.split("/")
                            if key0 == article_id:
                                s = articles[key]
                                # line = "[%s] %s" % (key1, s) # DEBUG
                                line = s
                                f.write("%s\n" % line.encode("utf-8"))


