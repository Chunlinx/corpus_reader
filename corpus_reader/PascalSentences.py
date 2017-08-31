# -*- coding: utf-8 -*-

import os
import re
import urllib2

import pyprind


class PascalSentences(object):
    """
    Pascal Sentences dataset
    """
    
    def __init__(self, path_images):
        # html text
        url = "http://vision.cs.uiuc.edu/pascal-sentences"
        if not os.path.exists("./pascal_sentences_html.txt"):
            htmltext = urllib2.urlopen(url).read()
            f = open("./pascal_sentences_html.txt", "w")
            f.writelines(htmltext)
            f.close()
        htmltext = open("./pascal_sentences_html.txt").readlines()

        # aggregation
        image_name_to_captions = {}
        re_img = re.compile('<img src="(.*?)">')
        re_cap = re.compile('<tr><td>(.*?)</td></tr>')
        cur_key = None
        for line in htmltext:
            line = line.strip()
            image_name = re_img.findall(line)
            cap = re_cap.findall(line)
            if len(image_name) != 0:
                image_name = image_name[0]
                image_name_to_captions[image_name] = []
                cur_key = image_name
            elif len(cap) != 0:
                cap = cap[0]
                cap = cap.strip()
                image_name_to_captions[cur_key].append(cap)
        
        # saving
        self.image_names = []
        self.captions = []
        if not os.path.exists(path_images):
            os.makedirs(path_images)
        for image_name in pyprind.prog_bar(image_name_to_captions.keys()):
            # save pairs [(image_name, caption1), (image_name, caption2), ...]
            caps = image_name_to_captions[image_name]
            for cap in caps:
                self.image_names.append(image_name)
                self.captions.append(cap)
            # save an image
            img = urllib2.urlopen(os.path.join(url, image_name))
            category = image_name.split("/")[0]
            if not os.path.exists(os.path.join(path_images, category)):
                os.makedirs(os.path.join(path_images, category))
            f = open(os.path.join(path_images, image_name), "wb")
            f.write(img.read())
            img.close()
            f.close()
        
    def get(self):
        return self.image_names, self.captions

    def dump(self, path_dir):
        image_names, captions = self.get()
        f_nam = open(os.path.join(
                        path_dir, "image_names.txt"), "w")
        f_cap = open(os.path.join(
                        path_dir, "captions.txt"), "w")
        for i in xrange(len(image_names)):
            f_nam.write("%s\n" % image_names[i].encode("utf-8"))
            f_cap.write("%s\n" % captions[i].encode("utf-8"))
        f_nam.flush()
        f_nam.close()
        f_cap.flush()
        f_cap.close()
