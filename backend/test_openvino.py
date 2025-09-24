# -*- coding: utf-8 -*-
import time
import cv2
import numpy as np
import os
import argparse
from openvino import Core

from src import craft_utils, imgproc, file_utils


def detect_openvino(compiled_model, input_name, image, text_threshold, link_threshold, low_text, poly=False):
    t0 = time.time()
    
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1 / target_ratio

    print("resize: {:.3f}".format(time.time() - t0))
    t0 = time.time()

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = np.transpose(x, (2, 0, 1))    # [h, w, c] -> [c, h, w]
    x = np.expand_dims(x, axis=0)     # [c, h, w] -> [b, c, h, w]
    x = x.astype(np.float32)

    print("preprocessing: {:.3f}".format(time.time() - t0))
    t0 = time.time()

    # forward pass OpenVINO
    results = compiled_model([x])  # inference
    # Récupérer la sortie (OpenVINO retourne un dict ou list)
    y_ov = results[compiled_model.outputs[0]]

    print("forward pass OpenVINO: {:.3f}".format(time.time() - t0))
    t0 = time.time()

    score_text = y_ov[0, :, :, 0]
    score_link = y_ov[0, :, :, 1]

    print("make score and link map: {:.3f}".format(time.time() - t0))
    t0 = time.time()

    # post-processing
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    print("Post-processing: {:.3f}".format(time.time() - t0))
    t0 = time.time()

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: 
            polys[k] = boxes[k]

    print("coordinate adjustment: {:.3f}".format(time.time() - t0))
    t0 = time.time()

    # render results
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    print("render results: {:.3f}".format(time.time() - t0))

    return boxes, polys, ret_score_text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CRAFT Text Detector with OpenVINO")

    parser.add_argument(
        "--model", default="./models/craft_mlt_25k.xml",
        type=str, help="path to OpenVINO IR .xml file (weights .bin must be in same folder)"
    )
    parser.add_argument("--image", default="./data/imgs/img_001.jpg", type=str, help="path to input image")
    parser.add_argument("--result_folder", default="./result/openvino/", type=str, help="folder to save results")
    parser.add_argument("--text_threshold", default=0.7, type=float, help="text confidence threshold")
    parser.add_argument("--link_threshold", default=0.4, type=float, help="link confidence threshold")
    parser.add_argument("--low_text", default=0.4, type=float, help="text low-bound score")
    parser.add_argument("--poly", action="store_true", help="enable polygon type")

    args = parser.parse_args()

    ie = Core()

    model_ir = ie.read_model(model=args.model, weights=args.model.replace(".xml", ".bin"))

    compiled_model = ie.compile_model(model=model_ir, device_name="CPU",config={"PERFORMANCE_HINT": "LATENCY"})

    input_name = compiled_model.inputs[0].get_any_name()

    image = imgproc.loadImage(args.image)

    boxes, polys, score_text = detect_openvino(
        compiled_model=compiled_model,
        input_name=input_name,
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