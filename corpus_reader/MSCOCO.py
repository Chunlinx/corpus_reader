# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import scipy.io as sio
import h5py
import pyprind

N_TRAIN = 566747
N_VAL = 25010
N_TEST = 25010

class MSCOCO(object):
    """
    MS COCO dataset (Karpathy ver.)
    """

    def __init__(self, path, path_vgg_feats):
        vgg_feats = sio.loadmat(os.path.join(path, "..", "karpathy", "coco", "vgg_feats.mat"))
        vgg_feats = vgg_feats["feats"].transpose(1, 0)
        
        self.image_names = {"train": [], "val": [], "test": []}
        self.captions = {"train": [], "val": [], "test": []}
        hdf5File_train = h5py.File(os.path.join(path_vgg_feats, "vgg_feats.train.h5"), "w")
        hdf5File_val = h5py.File(os.path.join(path_vgg_feats, "vgg_feats.val.h5"), "w")
        hdf5File_test = h5py.File(os.path.join(path_vgg_feats, "vgg_feats.test.h5"), "w")
        hdf5data_train = hdf5File_train.create_dataset("data", shape=(N_TRAIN, 4096), dtype=np.float32) 
        hdf5data_val = hdf5File_val.create_dataset("data", shape=(N_VAL, 4096), dtype=np.float32) 
        hdf5data_test = hdf5File_test.create_dataset("data", shape=(N_TEST, 4096), dtype=np.float32) 

        with open(os.path.join(path, "..", "karpathy", "coco", "dataset.json")) as f:
            data = json.load(f)
            data = data["images"]
            for d in pyprind.prog_bar(data):
                img_name = d["filename"]
                split_name = d["filepath"]
                split = d["split"]
                imgid = int(d["imgid"])
                captions = d["sentences"]

                assert split_name in ["train2014", "val2014"]
                
                path_img = os.path.join(path, "images", split_name, img_name)
                feats = vgg_feats[imgid]

                if split == u"train" or split == u"restval":
                    for cap in captions:
                        cap = " ".join(cap["tokens"])
                        self.image_names["train"].append(path_img)
                        self.captions["train"].append(cap)
                        hdf5data_train[len(self.image_names["train"])-1,:] = feats
                elif split == u"val":
                    for cap in captions:
                        cap = " ".join(cap["tokens"])
                        self.image_names["val"].append(path_img)
                        self.captions["val"].append(cap)
                        hdf5data_val[len(self.image_names["val"])-1,:] = feats
                elif split == u"test":
                    for cap in captions:
                        cap = " ".join(cap["tokens"])
                        self.image_names["test"].append(path_img)
                        self.captions["test"].append(cap)
                        hdf5data_test[len(self.image_names["test"])-1,:] = feats
        
        hdf5File_train.flush()
        hdf5File_train.close()
        hdf5File_val.flush()
        hdf5File_val.close()
        hdf5File_test.flush()
        hdf5File_test.close()
        
        self.hdf5Files = {
                "train": h5py.File(os.path.join(path_vgg_feats, "vgg_feats.train.h5"), "r"),
                "val": h5py.File(os.path.join(path_vgg_feats, "vgg_feats.val.h5"), "r"),
                "test": h5py.File(os.path.join(path_vgg_feats, "vgg_feats.test.h5"), "r")}

    def get(self, split):
        return self.image_names[split], self.hdf5Files[split]["data"], self.captions[split]

    def dump(self, path_dir):
        for split in ["train", "val", "test"]:
            image_names, feats, captions = self.get(split)
            f_nam = open(os.path.join(path_dir,
                "image_names.%s.txt" % split), "w")
            f_cap = open(os.path.join(path_dir,
                "captions.%s.txt" % split), "w")
            for i in xrange(len(image_names)):
                f_nam.write("%s\n" % image_names[i].encode("utf-8"))
                f_cap.write("%s\n" % captions[i].encode("utf-8"))
            f_nam.flush()
            f_nam.close()
            f_cap.flush()
            f_cap.close()

