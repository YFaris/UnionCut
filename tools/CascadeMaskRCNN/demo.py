# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/demo/demo.py

import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.structures import Instances
from detectron2.utils.logger import setup_logger

from cutler_config import add_cutler_config

from predictor import VisualizationDemo
from rcnn import GeneralizedRCNN
from custom_cascade_rcnn import CustomCascadeROIHeads
import argparse

# constants
WINDOW_NAME = "CutLER detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_cutler_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Disable the use of SyncBN normalization when running on a CPU
    # SyncBN is not supported on CPU and can cause errors, so we switch to BN instead
    if cfg.MODEL.DEVICE == 'cpu' and cfg.MODEL.RESNETS.NORM == 'SyncBN':
        cfg.MODEL.RESNETS.NORM = "BN"
        cfg.MODEL.FPN.NORM = "BN"
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser(model_path):
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="./cascade_mask_rcnn_R_50_FPN_demo.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        default="/home/wzl/CutLER/cutler/demo/imgs/demo1.jpg",
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", model_path],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def load_model(model_path):
    args = get_parser(model_path).parse_args()
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    return demo


def visualization(masks):
    """
    visualize each component with different color
    :return:
    """
    # sort the masks by area in descending order
    masks = list(masks)
    masks.sort(key=lambda x: np.sum(x), reverse=True)
    masks = np.array(masks)
    import random
    # obtain the number of colours
    colour_num = masks.shape[0]
    # generate random colours along each channel
    R = random.sample(list(range(256)), colour_num)
    G = random.sample(list(range(256)), colour_num)
    B = random.sample(list(range(256)), colour_num)
    # initialize an empty tensor to represent the visualization map
    visual_map = np.zeros((masks.shape[1], masks.shape[2], 3))
    # colour each mask
    for i in range(colour_num):
        visual_map[masks[i, :, :] == 1] = [R[i], G[i], B[i]]
    # transfer the map into uint8
    visual_map = visual_map.astype(np.uint8)
    return visual_map


if __name__ == "__main__":
    model_path = "/home/wzl/CutLER/cutler/output/r1/seed1/model_0059999.pth"
    img_folder = "/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/"

    # load model
    model = load_model(model_path=model_path)

    image_paths = [os.path.join(img_folder, filename) for filename in os.listdir(img_folder)]

    for image_path in image_paths:
        # load the image
        img = read_image(image_path, format="BGR")
        # prediction
        predictions, visualized_output = model.run_on_image(img)
        # filter background prediction
        # predictions = fore_union_filter(predictions, fore_union)
        try:
            # visualization
            colorful_masks = visualization(predictions['instances'].pred_masks.cpu().numpy())
        except:
            continue
        cv2.imshow("visualization", visualized_output.get_image()[:, :, ::-1])
        if cv2.waitKey(0) == ord('q'):
            break  # press 'q' to quit
    cv2.destroyAllWindows()
