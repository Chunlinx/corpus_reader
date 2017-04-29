# -*- coding: utf-8 -*-

import corpus_reader

def main():
    msrp = corpus_reader.MSRP("/mnt/hdd/dataset/Microsoft-Research-Paraphrase-Corpus")
    sick = corpus_reader.SICK("/mnt/hdd/dataset/SICK/semeval2014-task1")
    snli = corpus_reader.SNLI("/mnt/hdd/dataset/Stanford-Natural-Language-Inference/snli_1.0")
    sst = corpus_reader.SST("/mnt/hdd/dataset/Stanford-Sentiment-Treebank/stanfordSentimentTreebank/trees")
    mscoco = corpus_reader.MSCOCO("/mnt/hdd1/dataset/COCO/coco", "./")

    print "Done."


if __name__ == "__main__":
    main()
