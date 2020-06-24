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

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--test_path',type=str)
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

IMAGE_RESIZE = 640

count = 1


def batch_infe(net,batch_list):
    global net_infe_time
    global post_process_time
    image_size_list = []
    img_resized_list = []
    img_raw_list = []
    img_name_list = []
    for i_image in batch_list:
        img_name_list.append(i_image)
        img = cv2.imread(i_image,cv2.IMREAD_COLOR)
        img_raw_list.append(img)
        image_size_list.append([img.shape[1], img.shape[0]])
        # img.shape : h ,w ,c
        img_resized = cv2.resize(img,(IMAGE_RESIZE,IMAGE_RESIZE))
        img_resized = np.float32(img_resized)
        img_resized -= (104, 117, 123)
        img_resized = img_resized.transpose(2, 0, 1)
        img_resized = torch.from_numpy(img_resized)
        img_resized_list.append(img_resized)
    infe_begin_time = time.time()
    imgs_resized = torch.stack(img_resized_list, dim=0)
    imgs_resized = imgs_resized.to(device)
    loc, conf, landms = net(imgs_resized)  # forward pass
    infe_end_time = time.time()
    net_infe_time += (infe_end_time - infe_begin_time)
    post_begin_time = time.time()
    prior_data = priors.data
    for i in range(len(batch_list)):
        i_boxes = decode(loc[i], prior_data, cfg['variance'])
        box_scale = torch.Tensor(image_size_list[i]*2)
        i_boxes = i_boxes.cpu() * box_scale
        i_boxes = i_boxes.numpy()

        i_landms = decode_landm(landms[i],prior_data,cfg['variance'])
        landms_scale = torch.Tensor(image_size_list[i] * 5)
        i_landms = i_landms.cpu() * landms_scale
        i_landms = i_landms.numpy()

        i_scores = conf[i].cpu().numpy()[:,1]

        # ignore low scores
        inds = np.where(i_scores > args.confidence_threshold)[0]
        i_boxes = i_boxes[inds]
        i_landms = i_landms[inds]
        i_scores = i_scores[inds]

        # keep top-K before NMS
        order = i_scores.argsort()[::-1][:args.top_k]
        i_boxes = i_boxes[order]
        i_landms = i_landms[order]
        i_scores = i_scores[order]

        # do NMS
        dets = np.hstack((i_boxes, i_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        i_landms = i_landms[keep]

        # keep top-K after NMS
        dets = dets[:args.keep_top_k, :]
        i_landms = i_landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, i_landms), axis=1)

        # show image
        if False:
            img_raw =img_raw_list[i]
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
            # save image
            global  count
            name = 'images/' + "img_"+str(count) + '.jpg'
            count += 1
            print(name)
            cv2.imwrite(name, img_raw)
    post_end_time = time.time()
    post_process_time += (post_end_time - post_begin_time )



if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()

    print('Finished loading model!')
    # print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    priorbox = PriorBox(cfg, image_size=(IMAGE_RESIZE, IMAGE_RESIZE))
    priors = priorbox.forward()
    priors = priors.to(device)

    resize = 1
    batch_size = 4
    save_file_path = ''

    net_infe_time = 0.0
    post_process_time = 0.0
    # testing begin
    test_image_list = os.listdir(args.test_path)
    if args.test_path[-1] == '/':
        save_path = args.test_path[:-1] + '_vis'
        save_file_path = args.test_path[:-1] + '_resultFile'
    else:
        save_path = args.test_path + '_vis'
        save_file_path = args.test_path + '_resultFile'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f_write = open(save_file_path, 'w')
    test_image_list = [os.path.join(args.test_path, i) for i in test_image_list]
    test_image_list = sorted(test_image_list) * 100

    batch_test_images_list = [test_image_list[i:i+batch_size] for i in range(0,len(test_image_list),batch_size)]

    for i ,batch_list in enumerate(batch_test_images_list):
        print(i)
        batch_infe(net,batch_list)

    print("images num : ",len(test_image_list))
    print("net_infe_time : ",net_infe_time)
    print("post_process_time : ",post_process_time)


