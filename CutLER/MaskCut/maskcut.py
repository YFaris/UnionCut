#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys
sys.path.append('../')
import argparse
import numpy as np
from tqdm import tqdm
import re
import datetime
import PIL
import PIL.Image as Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from pycocotools import mask
import pycocotools.mask as mask_util
from scipy import ndimage
from scipy.linalg import eigh
import json
import cv2
import dino
# modified by the author of UnionCut based on third_party/UnionSeg
sys.path.append('../')
sys.path.append('../third_party')
from UnionSeg.inference import UnionSegMaster
# bilateral_solver codes are modfied based on https://github.com/poolio/bilateral_solver/blob/master/notebooks/bilateral_solver.ipynb
# from TokenCut.unsupervised_saliency_detection.bilateral_solver import BilateralSolver, BilateralGrid
# crf codes are are modfied based on https://github.com/lucasb-eyer/pydensecrf/blob/master/pydensecrf/tests/test_dcrf.py
from crf import densecrf


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8)*255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def create_image_info(image_id, file_name, image_size, 
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    """Return image_info in COCO style
    Args:
        image_id: the image ID
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        date_captured: the date this image info is created
        license: license of this image
        coco_url: url to COCO images if there is any
        flickr_url: url to flickr if there is any
    """
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }
    return image_info


def create_annotation_info(annotation_id, image_id, category_info, binary_mask, 
                           image_size=None, bounding_box=None):
    """Return annotation info in COCO style
    Args:
        annotation_id: the annotation ID
        image_id: the image ID
        category_info: the information on categories
        binary_mask: a 2D binary numpy array where '1's represent the object
        file_name: the file name of each image
        image_size: image size in the format of (width, height)
        bounding_box: the bounding box for detection task. If bounding_box is not provided, 
        we will generate one according to the binary mask.
    """
    upper = np.max(binary_mask)
    lower = np.min(binary_mask)
    thresh = upper / 2.0
    binary_mask[binary_mask > thresh] = upper
    binary_mask[binary_mask <= thresh] = lower
    if image_size is not None:
        binary_mask = resize_binary_mask(binary_mask.astype(np.uint8), image_size)

    binary_mask_encoded = mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))

    area = mask.area(binary_mask_encoded)
    if area < 1:
        return None

    if bounding_box is None:
        bounding_box = mask.toBbox(binary_mask_encoded)

    rle = mask_util.encode(np.array(binary_mask[...,None], order="F", dtype="uint8"))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    segmentation = rle

    annotation_info = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_info["id"],
        "iscrowd": 0,
        "area": area.tolist(),
        "bbox": bounding_box.tolist(),
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    } 

    return annotation_info

# necessay info used for coco style annotations
INFO = {
    "description": "ImageNet-1K: pseudo-masks with MaskCut",
    "url": "https://github.com/facebookresearch/CutLER",
    "version": "1.0",
    "year": 2023,
    "contributor": "Xudong Wang",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Apache License",
        "url": "https://github.com/facebookresearch/CutLER/blob/main/LICENSE"
    }
]

# only one class, i.e. foreground
CATEGORIES = [
    {
        'id': 1,
        'name': 'fg',
        'supercategory': 'fg',
    },
]

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]

output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []}

category_info = {
    "is_crowd": 0,
    "id": 1
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser('MaskCut script')
    # default arguments
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--vit-arch', type=str, default='small', choices=['base', 'small'], help='which architecture')
    parser.add_argument('--vit-feat', type=str, default='k', choices=['k', 'q', 'v', 'kqv'], help='which features')
    parser.add_argument('--patch-size', type=int, default=16, choices=[16, 8], help='patch size')
    parser.add_argument('--nb-vis', type=int, default=20, choices=[1, 200], help='nb of visualization')
    parser.add_argument('--img-path', type=str, default=None, help='single image visualization')

    # additional arguments
    parser.add_argument('--dataset-path', type=str, default="/home/wzl/DataSet/imagenet/train/", help='path to the dataset')
    parser.add_argument('--tau', type=float, default=0.15, help='threshold used for producing binary graph')
    parser.add_argument('--num-folder-per-job', type=int, default=1, help='the number of folders each job processes')
    parser.add_argument('--job-index', type=int, default=0, help='the index of the job (for imagenet: in the range of 0 to 1000/args.num_folder_per_job-1)')
    parser.add_argument('--fixed_size', type=int, default=480, help='rescale the input images to a fixed size')
    parser.add_argument('--pretrain_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--N', type=int, default=3, help='the maximum number of pseudo-masks per image')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--use-cupy', type=bool, default=False, help='use cupy for NCut')

    args = parser.parse_args()

    if args.pretrain_path is not None:
        url = args.pretrain_path
    if args.vit_arch == 'base' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        feat_dim = 768
    elif args.vit_arch == 'small' and args.patch_size == 8:
        if args.pretrain_path is None:
            url = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        feat_dim = 384

    # initialize UnionSeg model
    UnionSegInstance = UnionSegMaster.load_pretrained_UnionSeg()
    DinoInstance = UnionSegMaster.load_pretrained_DINO(patch_size=8, scale="base")
    if args.cpu:
        UnionSegInstance.cpu()
        DinoInstance.cpu()

    img_folders = os.listdir(args.dataset_path)

    if args.out_dir is not None and not os.path.exists(args.out_dir) :
        os.mkdir(args.out_dir)

    start_idx = max(args.job_index*args.num_folder_per_job, 0)
    end_idx = min((args.job_index+1)*args.num_folder_per_job, len(img_folders))

    image_id, segmentation_id = 1, 1
    image_names = []
    for step, img_folder in enumerate(img_folders[start_idx:end_idx]):
        args.img_dir = os.path.join(args.dataset_path, img_folder)
        img_list = sorted(os.listdir(args.img_dir))

        for img_name in tqdm(img_list, desc=str(start_idx) + "->" + str(end_idx) + "| folder: " + str(step)):
            # get image path
            img_path = os.path.join(args.img_dir, img_name)
            # load a test image
            image = cv2.imread(img_path)
            try:
                # get pseudo-masks for each image using TokenCut + UnionSeg
                FeatureMaster = UnionSegMaster(image, tau=args.tau, eps=1e-5, UnionSegPatchSize=8, NCutPatchSize=8,
                                               UnionSegResize=224,
                                               NCutResize=args.fixed_size, device="cpu", pretrained_dino=DinoInstance,
                                               pretrained_UnionSeg=UnionSegInstance)
                _, _, ks, _ = FeatureMaster._DINO_features()
                ks = ks[1:, :]
                raw_fore_union = FeatureMaster.returnUnion()
                _, cos_similarity_matrix = FeatureMaster._square_distance_matrix(
                    FeatureMaster._norm_features(ks),
                    distance_mode="Euclidean",
                    feat_nums=ks.shape[0])
                TokenUnionSeg = UnionSegMaster(image, tau=args.tau, eps=1e-5, UnionSegPatchSize=8, NCutPatchSize=8,
                                               UnionSegResize=224,
                                               NCutResize=args.fixed_size, device="cpu", UnionSegOutput=raw_fore_union,
                                               NCutFeatures=ks, NCutCosSimilarityMatrix=cos_similarity_matrix, use_cupy=args.use_cupy)
                masks, fore_union = TokenUnionSeg.maskcut_segmentation(vis=False)
                if masks.shape[0] == 0:
                    print(f'Skipping {img_name}')
                    continue
            except:
                print(f'Skipping {img_name}')
                continue

            for idx, pseudo_mask in enumerate(masks):
                pseudo_mask = pseudo_mask.astype(np.uint8)
                pseudo_mask = cv2.resize(pseudo_mask, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
                pseudo_mask = pseudo_mask * 255

                # create coco-style image info
                if img_name not in image_names:
                    image_info = create_image_info(
                        image_id, "{}/{}".format(img_folder, img_name), (pseudo_mask.shape[0], pseudo_mask.shape[1], 3))
                    output["images"].append(image_info)
                    image_names.append(img_name)           

                # create coco-style annotation info
                annotation_info = create_annotation_info(
                    segmentation_id, image_id, category_info, pseudo_mask.astype(np.uint8), None)
                if annotation_info is not None:
                    output["annotations"].append(annotation_info)
                    segmentation_id += 1
            image_id += 1

    # save annotations
    if len(img_folders) == args.num_folder_per_job and args.job_index == 0:
        json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N)
    else:
        json_name = '{}/imagenet_train_fixsize{}_tau{}_N{}_{}_{}.json'.format(args.out_dir, args.fixed_size, args.tau, args.N, start_idx, end_idx)
    with open(json_name, 'w') as output_json_file:
        json.dump(output, output_json_file)
    print(f'dumping {json_name}')
    print("Done: {} images; {} anns.".format(len(output['images']), len(output['annotations'])))
