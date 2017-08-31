# -*- coding: utf-8 -*-

import corpus_reader

def main():
    print "Loading MSRP corpus ..."
    msrp = corpus_reader.MSRP("/mnt/hdd/dataset/Microsoft-Research-Paraphrase-Corpus")
    print "Done."
    
    print "Loading SICK corpus ..."
    sick = corpus_reader.SICK("/mnt/hdd/dataset/SICK/semeval2014-task1")
    print "Done."

    print "Loading SNLI corpus ..."
    snli = corpus_reader.SNLI("/mnt/hdd/dataset/Stanford-Natural-Language-Inference/snli_1.0")
    print "Done."

    print "Loading SST corpus ..."
    sst = corpus_reader.SST("/mnt/hdd/dataset/Stanford-Sentiment-Treebank/stanfordSentimentTreebank/trees")
    print "Done."

    print "Loading bAbI corpus ..."
    babi = corpus_reader.BABI("/mnt/hdd/dataset/bAbI/tasks_1-20_v1-2/en/", task_id=3)
    print "Done."
    
    print "Loading CBT corpus ..."
    cbt = corpus_reader.CBT("/mnt/hdd/dataset/ChildrensBookTest/CBTest", word_type="NE")
    print "Done."

    print "Loading MSCOCO dataset ..."
    mscoco = corpus_reader.MSCOCO("/mnt/hdd1/dataset/COCO/coco", path_vgg_feats="./")
    print "Done."

    print "Loading Pascal Sentences dataset ..."
    pascal = corpus_reader.PascalSentences(path_images="./pascal_sentences/images")
    print "Done."

if __name__ == "__main__":
    main()
