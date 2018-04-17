# -*- coding: utf-8 -*-

import bz2
from collections import OrderedDict
import os
import re

import numpy as np

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
        for src in ["lat", "nyt", "reu", "wsj", "noname"]:
            if not os.path.exists(os.path.join(path_dir, src)):
                os.makedirs(os.path.join(path_dir, src))
        src_map = {
                "lat": "lat",
                "nyt": "nyt",
                "reu": "reu",
                "199": "wsj", # 199*で始まるファイルはWSJ
                "noname": "noname"}
        output_id_map = {
                "lat": -1,
                "nyt": -1,
                "reu": -1,
                "199": -1,
                "noname": -1}
        cur_sentence_index = np.inf
        cur_article_name = None

        for dir_i, dir_name in enumerate(pyprind.prog_bar(dir_list)):
            # Directories that contain error files: 271, 297, 299
            if int(dir_name) in [271, 297, 299]:
                print("Skipped.")
                continue

            file_names = os.listdir(os.path.join(self.path, dir_name))
            file_names.sort()
            n_files = len(file_names)

            for file_i, file_name in enumerate(file_names):
                print("[%d/%d, %d/%d] %s ..." % (dir_i+1, n_dirs, file_i+1, n_files, os.path.join(self.path, dir_name, file_name)))
                # Read 
                bz_file = bz2.BZ2File(os.path.join(self.path, dir_name, file_name))
                lines = bz_file.readlines()
                cache = OrderedDict()
                for line in lines:
                    # line = line.decode("latin8").strip()
                    line = line.strip()
                    if re_head.match(line):
                        # 新しい文への切り替え点
                        match = re_head.findall(line)
                        match = match[0]
                        # n_parses = int(match[0])
                        article_name = match[1]
                        if article_name == "":
                            article_head = "noname"
                        else:
                            article_head = article_name[:3]
                        output_dir = src_map[article_head]
                        sentence_index = int(match[2])
                        
                        # もしsentence_indexが振り出しに戻ったら(e.g., 0 or 1), 新しいファイル名とする
                        # あるいはarticle_nameが変わったら, 新しいファイル名とする
                        if sentence_index < cur_sentence_index:
                            output_id_map[article_head] += 1
                            output_name = "%06d.txt" % output_id_map[article_head]
                        elif article_name != cur_article_name:
                            output_id_map[article_head] += 1
                            output_name = "%06d.txt" % output_id_map[article_head]
                        else:
                            output_name = "%06d.txt" % output_id_map[article_head]
                        cur_sentence_index = sentence_index
                        cur_article_name = article_name

                        cache["%s/%d" % (output_name, sentence_index)] = False
                    elif re_tree.match(line):
                        # 文. まだassignしてなければ読み込み，既にしてればスルー.
                        if not cache["%s/%d" % (output_name, sentence_index)]:
                            # まだassignしてなければ読み込む
                            tree = bllipparser.Tree(line)
                            tokens = tree.tokens()
                            raw_sent = " ".join(tokens)
                            raw_sent = raw_sent.replace("-LRB-", "(").replace("-RRB-", ")")
                            # Write
                            with open(os.path.join(path_dir, output_dir, output_name), "a") as f:
                                sent = raw_sent.decode("latin8")
                                sent = sent.encode("utf-8")
                                if self.debug_mode:
                                    sent = "[%s] %s" % (sentence_index, sent)
                                f.write("%s\n" % sent)
                            cache["%s/%d" % (output_name, sentence_index)] = True
                        else:
                            # 既にassignしてれば何もしない (計算コストの節約)
                            continue
                    else:
                        # 恐らくparseスコア
                        continue

