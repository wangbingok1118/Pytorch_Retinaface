from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time

confidence_threshold = 0.02
nms_threshold = 0.4
top_k = 5000
keep_top_k = 750
cfg = cfg_re50
resize = 1
priorbox = PriorBox(cfg, image_size=(640, 640))
priors = priorbox.forward()
priors = priors.to('cuda')

def postprocess_detection(device,loc_batch,conf_batch,landms_batch):
    dets_batch = []
    for i in range(loc_batch.shape[0]):
        i_loc = loc_batch[i]
        i_conf = conf_batch[i]
        i_landms = landms_batch[i]

        i_loc = torch.from_numpy(i_loc).to(device)
        i_conf = torch.from_numpy(i_conf).to(device)
        i_landms = torch.from_numpy(i_landms).to(device)
        prior_data = priors.data

        scale = torch.Tensor([640]*4)
        scale = scale.to(device)
        boxes = decode(i_loc, prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = i_conf.cpu().numpy()[:, 1]

        landms = decode_landm(i_landms, prior_data, cfg['variance'])
        scale1 = torch.Tensor([640]*10)
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()
        # print("detect done")
        # print(scores.shape)
        # print(boxes.shape)
        # print(landms.shape)

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        # print("ignore low scores")
        # print(scores.shape)
        # print(boxes.shape)
        # print(landms.shape)

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        # print("keep top-K before NMS")
        # print(scores.shape)
        # print(boxes.shape)
        # print(landms.shape)

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        # print("do NMS")
        # print(dets.shape)
        # print(landms.shape)

        # keep top-K after NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        # print("keep top-K aftWer NMS")
        # print(dets.shape)
        # print(landms.shape)


        dets = np.concatenate((dets, landms), axis=1)
        dets_batch.append(dets)
    return dets_batch

