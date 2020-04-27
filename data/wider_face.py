import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from  collections  import OrderedDict
import json

class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        self.image_folder = txt_path.replace('.csv','')
        image_bbox_dict = OrderedDict()
        with open(txt_path,'r') as f:
            for line in f.readlines()[:100]:
                line = line.strip()
                if not line:
                    continue
                image_name = line.split(',')[0]
                bbox = [float(x) for x in  line.split(',')[1:]] # iqi_cartoon face x1,y1,x2,y2
                if image_name not in image_bbox_dict:
                    image_bbox_dict[image_name] = []
                image_bbox_dict[image_name].append(bbox)
        for k,v in image_bbox_dict.items():
            self.imgs_path.append(os.path.join(self.image_folder,k))
            self.words.append(v)
        image_bbox_dict.clear()
    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 5))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 5))
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[2]  # x2
            annotation[0, 3] = label[3]  # y2
            annotation[0, 4] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
