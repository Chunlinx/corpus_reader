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
        re_head = re.compile(r"([0-9]+) (.*)_([0-9]+)")
        re_tree = re.compile(r"^\(.*\)$")
        dir_list = os.listdir(self.path)
        dir_list.sort()
        for dir_name in dir_list:
            print("Processing %s ..." % os.path.join(self.path, dir_name))
            # Error files: 271, 297, 299
            if int(dir_name.split(".")[0]) in [271, 297, 299]:
                print("Skipped.")
                continue

            f = open(os.path.join(path_dir, dir_name + ".txt"), "w")
            file_names = os.listdir(os.path.join(self.path, dir_name))
            file_names.sort()
            for file_name in pyprind.prog_bar(file_names):
                # Read lines
                path_file = os.path.join(self.path, dir_name, file_name)
                bz_file = bz2.BZ2File(path_file)
                lines = bz_file.readlines()
                # Aggregate raw sentences
                contents = OrderedDict()
                for line in lines:
                    # line = line.strip()
                    line = line.decode("latin8").strip()
                    if re_head.match(line):
                        match = re_head.findall(line)
                        match = match[0]
                        # n_parses = int(match[0])
                        article_id = match[1]
                        sent_number = int(match[2])
                        contents["%s %d" % (article_id, sent_number)] = None
                    elif re_tree.match(line):
                        tree = bllipparser.Tree(line)
                        tokens = tree.tokens()
                        raw_sent = " ".join(tokens)
                        raw_sent = raw_sent.replace("-LRB-", "(").replace("-RRB-", ")")
                        contents["%s %d" % (article_id, sent_number)] = raw_sent
                    else:
                        continue
                # Wtite
                for key in contents.keys():
                    f.write("%s\n" % contents[key].encode("utf-8"))
                f.flush()
            f.close()


