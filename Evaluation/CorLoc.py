import os
from pycocotools.coco import COCO
import h5py
from tqdm import tqdm
import cv2
import numpy as np
import xml.dom.minidom
import torch
import argparse


def find_bbox(binary_image, mass_center=True):
    """
    Find the bounding box of the largest object in a binary image.

    :param binary_image: 2D numpy array of shape (H, W).
    :return: tuple of coordinates (x, y, w, h). x and y are the mass center or center of the bbox
    """
    # obtain indices of mask pixels
    ys, xs = np.where(binary_image == 1)

    # obtain coordinates of left top and right bottom corner
    ymin, xmin = np.min(ys), np.min(xs)
    ymax, xmax = np.max(ys), np.max(xs)

    # obtain height and width of the bbox
    h, w = ymax - ymin, xmax - xmin

    if mass_center:
        # obtain mass center of the instance
        # obtain shape of the mask
        row, col = binary_image.shape
        # obtain meshgrid
        coord_xs, coord_ys = np.meshgrid(np.arange(0, col, 1), np.arange(0, row, 1))
        coord = np.stack([coord_xs, coord_ys], axis=0)
        # calculate mass center
        x_mass, y_mass = np.sum((binary_image * coord).reshape(2, -1), axis=1) / np.sum(binary_image)
        return [x_mass, y_mass, w, h]
    else:
        # return normal bbox
        return [xmin, ymin, xmax, ymax]


def IoUbox(rect1, rect2):
    """
    calculate IOU between two bbox
    params：
        rect1:[left1, top1, right1, bottom1]
        rect2:[left2, top2, right2, bottom2]
    return：
        iou
    """
    left1 = rect1[0]
    top1 = rect1[1]
    right1 = rect1[2]
    bottom1 = rect1[3]
    left2 = rect2[0]
    top2 = rect2[1]
    right2 = rect2[2]
    bottom2 = rect2[3]

    s_rect1 = (bottom1 - top1) * (right1 - left1)
    s_rect2 = (bottom2 - top2) * (right2 - left2)

    cross_left = max(left1, left2)
    cross_right = min(right1, right2)
    cross_top = max(top1, top2)
    cross_bottom = min(bottom1, bottom2)

    if cross_left >= cross_right or cross_top >= cross_bottom:
        return 0
    else:
        s_cross = (cross_right - cross_left) * (cross_bottom - cross_top)
        return s_cross / (s_rect1 + s_rect2 - s_cross)


def get_bboxes(xml_path):
    # load the xml file
    dom = xml.dom.minidom.parse(xml_path)
    # obtain the root of the xml file
    root = dom.documentElement
    # obtain all objects blocks
    objects = root.getElementsByTagName("object")
    # obtain the name of the image
    image_name = root.getElementsByTagName('filename')[0].childNodes[0].data
    # initialize the list for bboxes of the image
    bboxes = []
    # obtain all boxes
    for obj in objects:
        # print ("*****Object*****")
        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0]
        xmin_data = int(float(xmin.childNodes[0].data))
        # print(xmin_data)
        ymin = bndbox.getElementsByTagName('ymin')[0]
        ymin_data = int(float(ymin.childNodes[0].data))
        # print(ymin_data)
        xmax = bndbox.getElementsByTagName('xmax')[0]
        xmax_data = int(float(xmax.childNodes[0].data))
        # print(xmax_data)
        ymax = bndbox.getElementsByTagName('ymax')[0]
        ymax_data = int(float(ymax.childNodes[0].data))
        bboxes.append([xmin_data, ymin_data, xmax_data, ymax_data])
    return image_name, bboxes


def VOC_CorLoc(h5_path, method, xml_root):
    """
    calculate CorLoc for VOC2007 & VOC2012
    h5_path: the path of the h5 file saving the pseudo-label
    method: the UOD method name
    xml_root: the path saving xml label files
    """
    # load the h5 file
    file = h5py.File(h5_path, "r")
    method_group = file[method]
    # initialize image names of the dataset
    xml_names = list(method_group.keys())
    # initialize the total number of valid images
    counter = 0
    # initialize the number of matched images
    match = 0
    # initialize list for saving correct matched images
    matched_images = []
    # traverse each xml file
    for xml_name in tqdm(xml_names):
        # initialize the path of the xml file
        xml_path = xml_root + xml_name + ".xml"
        # obtain the image name and ground truth bboxes of it
        image_name, gt_bboxes = get_bboxes(xml_path)
        # obtain the masks of the image
        try:
            # load the prediction of UOD
            predicted_masks = method_group[image_name.split('.')[0] + '/masks'][0:1]
        except:
            continue
        # obtain the bboxes of the predicted masks
        predicted_bboxes = []
        for i in range(predicted_masks.shape[0]):
            predicted_bboxes.append(find_bbox(predicted_masks[i], mass_center=False))
        # compare the predicted bboxes and ground truth bboxes
        correct = False
        for p_bbox in predicted_bboxes:
            for gt_bbox in gt_bboxes:
                if IoUbox(p_bbox, gt_bbox) >= 0.5:
                    correct = True
                    break
            if correct:
                break
        # update counter
        if correct:
            match += 1
            matched_images.append(xml_name)
        counter += 1
    file.close()
    return match / counter, matched_images


def COCO20K_CorLoc(h5_path, method, annotations_path):
    """
    calculate CorLoc for COCO20K
    h5_path: the path of the h5 file saving the pseudo-label
    method: the UOD method name
    annotations_path: the json path of annotations
    """
    # load the h5 file
    file = h5py.File(h5_path, "r")
    method_group = file[method]
    # initialize image names of the dataset
    image_names = list(method_group.keys())
    # initialize COCO api for instance annotations
    coco = COCO(annotations_path)
    # initialize the total number of valid images
    counter = 0
    # initialize the number of matched images
    match = 0
    # initialize list for saving correct matched images
    matched_images = []
    # load each image's annotation
    for image_name in tqdm(image_names):
        # obtain the image id
        image_id = int(image_name.split('.')[0].split('_')[-1])
        # obtain the annotation id
        annotation_id = coco.getAnnIds(imgIds=[image_id])
        # obtain the annotations of the image
        annotations = coco.loadAnns(annotation_id)
        # obtain the masks of the image
        try:
            # load the prediction of UOD
            predicted_masks = method_group[image_name.split('.')[0] + '/masks'][0:1]
            # obtain the bboxes of the predicted masks
            predicted_bboxes = []
            for i in range(predicted_masks.shape[0]):
                predicted_bboxes.append(find_bbox(predicted_masks[i], mass_center=False))
        except:
            continue
        # compare the predicted bboxes and ground truth bboxes
        correct = False
        for p_bbox in predicted_bboxes:
            # compare the prediction with each annotation of the image
            for annotation in annotations:
                # load the bbox
                bbox = annotation['bbox']
                # analyse the bbox
                xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
                if IoUbox(p_bbox, [xmin, ymin, xmax, ymax]) >= 0.5:
                    correct = True
                    break
            if correct:
                break
        # update counter
        if correct:
            match += 1
            matched_images.append(image_name)
        counter += 1
    file.close()
    return match / counter, matched_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser('CorLoc Evaluation')
    # default arguments
    parser.add_argument('--h5-path', type=str,
                        default="/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                        help='path of the pre-generated h5 file of UOD results')
    parser.add_argument('--uod-method', type=str, default="UnionSeg", choices=["UnionSeg", "CutLER"],
                        help='UOD method used to generate the h5 file')
    parser.add_argument('--dataset', type=str, default="VOC", choices=["VOC", "COCO20K"], help='name of the benchmark')
    parser.add_argument('--gt-path', type=str, default="/home/wzl/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/Annotations/",
                        help='path of the ground truth folder/file')
    args = parser.parse_args()

    if args.dataset == "VOC":
        corloc, _ = VOC_CorLoc(h5_path=args.h5_path,
                               method=args.uod_method,
                               xml_root=args.gt_path)
        print("VOC CorLoc:", corloc)
    elif args.dataset == "COCO20K":
        corloc, _ = COCO20K_CorLoc(
            h5_path=args.h5_path,
            method=args.uod_method,
            annotations_path=args.gt_path)
        print("COCO20K CorLoc:", corloc)