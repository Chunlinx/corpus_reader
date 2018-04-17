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

    def __init__(self, path, debug_mode=False):
        self.path = path
        self.debug_mode = debug_mode

    def dump(self, path_dir):
        # e.g., 50 reute9406_007.0356_13
        # 50 => # of parses
        # reute9406_007 =>  article id from the North Americal News Text Corpus
        # 13 => the index of the sentence in the article
        re_head = re.compile(r"([0-9]+) (.*)_([0-9]+)")
        re_tree = re.compile(r"^\(.*\)$")
        dir_list = os.listdir(self.path)
        dir_list.sort()
        n_dirs = len(dir_list)
        
        # source materialごとの保存先ディレクトリ作成
        for src in ["lat", "nyt", "reu", "199", "noname"]:
            if not os.path.exists(os.path.join(path_dir, src)):
                os.makedirs(os.path.join(path_dir, src))
        src_map = {w:w for w in ["lat", "nyt", "reu", "199"]}

        for dir_i, dir_name in enumerate(pyprind.prog_bar(dir_list)):
            # Directories that contain error files: 271, 297, 299
            if int(dir_name) in [271, 297, 299]:
                print("Skipped.")
                continue

            file_names = os.listdir(os.path.join(self.path, dir_name))
            file_names.sort()
            n_files = len(file_names)
            noname_count = 0
            for file_i, file_name in enumerate(file_names):
                print("[%d/%d, %d/%d] %s ..." % (dir_i+1, n_dirs, file_i+1, n_files, os.path.join(self.path, dir_name, file_name)))
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
                    if article_id != "":
                        output_name = article_id + ".txt"
                        output_dir = src_map.get(article_id[:3], "noname")
                    else:
                        # in the case of dir_name in ["484", 485", "486", "487", "488", "489", "490", "491", "492", "493", "494", "495", "496", "497", "498", "499", "500", "501", "502", "503"]
                        output_name = "%s.02d.txt" % (dir_name, noname_count)
                        noname_count += 1
                        output_dir = "noname"
                    with open(os.path.join(path_dir, output_dir, output_name), "a") as f:
                        for key in articles:
                            key0, key1 = key.split("/")
                            if key0 == article_id:
                                s = articles[key]
                                if self.debug_mode:
                                    line = "[%s] %s" % (key1, s) # DEBUG
                                else:
                                    line = s
                                f.write("%s\n" % line.encode("utf-8"))


