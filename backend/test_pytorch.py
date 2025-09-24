# -*- coding: utf-8 -*-
import os
import time
import torch
from torch.autograd import Variable
import cv2
import numpy as np
from collections import OrderedDict
import argparse

from src import craft_utils, imgproc, file_utils
from src.craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def detect(net, image, text_threshold, link_threshold, low_text, poly):
    t0 = time.time()
    
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5)
    ratio_h = ratio_w = 1 / target_ratio

    t1 = time.time()
    temps_total = t1 - t0
    print("resize: {:.3f}".format(temps_total))

    t0 = time.time()

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    
    t1 = time.time()
    temps_total = t1 - t0
    print("preprocessing: {:.3f}".format(temps_total))

    t0 = time.time()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)
    
    t1 = time.time()
    temps_total = t1 - t0
    print("forward pass: {:.3f}".format(temps_total))


    t0 = time.time()
    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    t1 = time.time()
    temps_total = t1 - t0
    print("make score and link map: {:.3f}".format(temps_total))

    t0 = time.time()
    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    t1 = time.time()
    temps_total = t1 - t0
    print("Post-processing: {:.3f}".format(temps_total))


    t0 = time.time()
    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time()
    temps_total = t1 - t0
    print("coordinate adjustment: {:.3f}".format(temps_total))

    t0 = time.time()
    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    t1 = time.time()
    temps_total = t1 - t0
    print("render results: {:.3f}".format(temps_total))

    return boxes, polys, ret_score_text

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CRAFT Text Detector with OpenVINO")

    parser.add_argument("--model", default="./models/craft_mlt_25k.pth", type=str, help="path to trained model")
    parser.add_argument("--image", default="./data/imgs/img_001.jpg", type=str, help="path to input image")
    parser.add_argument("--result_folder", default="./result/pytorch/", type=str, help="folder to save results")
    parser.add_argument("--text_threshold", default=0.7, type=float, help="text confidence threshold")
    parser.add_argument("--link_threshold", default=0.4, type=float, help="link confidence threshold")
    parser.add_argument("--low_text", default=0.4, type=float, help="text low-bound score")
    parser.add_argument("--poly", action="store_true", help="enable polygon type")

    args = parser.parse_args()

    net = CRAFT()
    net.load_state_dict(copyStateDict(torch.load(args.model, map_location='cpu')))
    net.eval()

    image = imgproc.loadImage(args.image)
    bboxes, polys, score_text = detect(
        net=net,
        image=image,
        text_threshold=args.text_threshold,
        link_threshold=args.link_threshold,
        low_text=args.low_text,
        poly=args.poly
    )

    filename, file_ext = os.path.splitext(os.path.basename(args.image))
    mask_file = os.path.join(args.result_folder, f"res_{filename}_mask.jpg")
    cv2.imwrite(mask_file, score_text)

    file_utils.saveResult(args.image, image[:,:,::-1], polys, dirname=args.result_folder)