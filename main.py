# -*- coding: utf-8 -*-

import reader

def main():
    msrp = reader.MSRP("/mnt/hdd/dataset/Microsoft-Research-Paraphrase-Corpus")
    sick = reader.SICK("/mnt/hdd/dataset/SICK/semeval2014-task1")
    snli = reader.SNLI("/mnt/hdd/dataset/Stanford-Natural-Language-Inference/snli_1.0")
    sst = reader.SST("/mnt/hdd/dataset/Stanford-Sentiment-Treebank/stanfordSentimentTreebank/trees")

    print "Done."


if __name__ == "__main__":
    main()
