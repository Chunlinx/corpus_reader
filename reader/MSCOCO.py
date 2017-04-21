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
    Karpathy ver.
    """
    def __init__(self, path, path_vgg_feats):
        self.load_dataset(path, path_vgg_feats)

    def load_dataset(self, path, path_vgg_feats):
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

    def get_data(self, split):
        return self.image_names[split], self.hdf5Files[split]["data"], self.captions[split]


# class MSCOCO(object):
#     def __init__(self, path):
#         sys.path.append(os.path.join(path, "PythonAPI"))
#         self.load_dataset(path)
#
#     def load_dataset(self, path):
#         self.images = {}
#         self.captions = {}
#         self.images["train"], self.captions["train"] = self.process_split(path, split="train2014")
#         self.images["val"], self.captions["val"] = self.process_split(path, split="val2014")
#
#     def process_split(self, path, split):
#         from pycocotools.coco import COCO
#         coco_ann = COCO(os.path.join(path, "annotations", "instances_%s.json" % split))
#         coco_cap = COCO(os.path.join(path, "annotations", "captions_%s.json" % split))
#        
#         cats = coco_ann.loadCats(coco_ann.getCatIds())
#         cat_names = [x["name"] for x in cats]
#         supercat_names = list(set([x["supercategory"] for x in cats]))
#         cat_names = cat_names + supercat_names
#
#         dataset = {} # (image name) -> (caption)
#         for cat_name in cat_names:
#             catIds = coco_ann.getCatIds(catNms=[cat_name])
#             imgIds = coco_ann.getImgIds(catIds=catIds)
#             img_info_list = coco_ann.loadImds(imgIds)
#
#             for img_info in img_info_list:
#                 # get image path
#                 path_img = os.path.join(path, "images", split, img_info["file_name"])
#
#                 # # img = cv2.imread(os.path.join(path, "images", split, img_info["file_name"]))
#                 # img = io.imread(os.path.join(path, "images", split, img_info["file_name"]))
#                 # plt.figure()
#                 # plt.imshow(img)
#                 # plt.show()
#
#                 annIds = coco_cap.getAnnIds(imgIds=img_info["id"])
#                 anns = coco_cap.loadAnns(annIds)
#                 caps = [ann[u"caption"] for ann in anns]
#                 if dataset.has_key(path_img):
#                     dataset[path_img].extend(caps)
#                 else:
#                     dataset[path_img] = caps
#
#         for path_img in dataset.keys():
#             dataset[path_img] = list(set(dataset[path_img]))
#             assert 1 <= len(dataset[path_img]) <= 5
#
#         self.images = []
#         self.captions = []
#         for path_img, caps in dataset.items():
#             self.images.append(path_img)
#             for cap in caps:
#                 self.images.append(path_img)
#                 self.captions.append(cap)
#
#     def get_data(self, split):
#         return self.images[split], self.captions[split]
