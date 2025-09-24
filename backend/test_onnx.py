# -*- coding: utf-8 -*-
import onnxruntime
import time
import cv2
import numpy as np
import os
import argparse

from src import craft_utils, imgproc, file_utils

def detect_onnx(ort_session, image, text_threshold, link_threshold, low_text, poly=False):
    t0 = time.time()
    
    # resize (identique)
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
        image, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    ratio_h = ratio_w = 1 / target_ratio

    t1 = time.time()
    print("resize: {:.3f}".format(t1 - t0))

    t0 = time.time()

    # preprocessing (identique mais sans PyTorch)
    x = imgproc.normalizeMeanVariance(img_resized)
    x = np.transpose(x, (2, 0, 1))    # [h, w, c] to [c, h, w]
    x = np.expand_dims(x, axis=0)     # [c, h, w] to [b, c, h, w]
    x = x.astype(np.float32)          # Conversion en float32

    t1 = time.time()
    print("preprocessing: {:.3f}".format(t1 - t0))

    t0 = time.time()

    # forward pass AVEC ONNX (remplace net(x))
    ort_inputs = {ort_session.get_inputs()[0].name: x}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    y_onnx = ort_outputs[0] 

    t1 = time.time()
    print("forward pass ONNX: {:.3f}".format(t1 - t0))

    t0 = time.time()

    score_text = y_onnx[0, :, :, 0]  # Region score
    score_link = y_onnx[0, :, :, 1]  # Affinity score

    t1 = time.time()
    print("make score and link map: {:.3f}".format(t1 - t0))

    t0 = time.time()

    # Post-processing (identique)
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    t1 = time.time()
    print("Post-processing: {:.3f}".format(t1 - t0))

    t0 = time.time()

    # coordinate adjustment (identique)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: 
            polys[k] = boxes[k]

    t1 = time.time()
    print("coordinate adjustment: {:.3f}".format(t1 - t0))

    t0 = time.time()

    # render results (identique)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    t1 = time.time()
    print("render results: {:.3f}".format(t1 - t0))

    return boxes, polys, ret_score_text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CRAFT Text Detector with OpenVINO")

    parser.add_argument("--model", default="./models/craft_mlt_25k.onnx", type=str, help="path to trained model")
    parser.add_argument("--image", default="./data/imgs/img_001.jpg", type=str, help="path to input image")
    parser.add_argument("--result_folder", default="./result/onnx/", type=str, help="folder to save results")
    parser.add_argument("--text_threshold", default=0.7, type=float, help="text confidence threshold")
    parser.add_argument("--link_threshold", default=0.4, type=float, help="link confidence threshold")
    parser.add_argument("--low_text", default=0.4, type=float, help="text low-bound score")
    parser.add_argument("--poly", action="store_true", help="enable polygon type")

    args = parser.parse_args()

    # Tester le modèle ONNX
    ort_session = onnxruntime.InferenceSession(args.model)

    # Charger votre image
    image = imgproc.loadImage(args.image)

    # Appeler la fonction ONNX
    boxes, polys, score_text = detect_onnx(
        ort_session=ort_session,
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