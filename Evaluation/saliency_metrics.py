import cv2
import numpy as np
import h5py
from tqdm import tqdm
import math
import argparse


def IoU(mask1, mask2):
    """
    calculate iou between mask1 and mask2
    mask1: (w, h) ndarray
    mask2: (w, h) ndarray
    """
    intersection = mask1 * mask2
    union = mask1 + mask2 - intersection
    return np.sum(intersection) / np.sum(union)


def Precision(mask, gt_mask):
    """
    calculate precision between mask1 and mask2
    mask: (w, h) ndarray
    gt_mask: (w, h) ndarray
    """
    intersection = mask * gt_mask
    return np.sum(intersection) / np.sum(mask)


def Recall(mask, gt_mask):
    """
    calculate precision between mask1 and mask2
    mask: (w, h) ndarray
    gt_mask: (w, h) ndarray
    """
    intersection = mask * gt_mask
    return np.sum(intersection) / np.sum(gt_mask)


def Accuracy(mask, gt_mask):
    """
    calculate accuracy of mask
    mask: (w, h) ndarray
    gt_mask: (w, h) ndarray
    """
    fore_intersection = mask * gt_mask
    back_intersection = (1 - mask) * (1 - gt_mask)
    return (np.sum(fore_intersection) + np.sum(back_intersection)) / (mask.shape[0] * mask.shape[1])


def F_max(mask, gt_mask, beta=0.3):
    """
    calculate max F beta of mask
    mask: (w, h) ndarray
    gt_mask: (w, h) ndarray
    """
    # calculate precision
    precision = Precision(mask, gt_mask)
    # calculate recall
    recall = Recall(mask, gt_mask)
    # calculate max F beta
    f_scores = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall)
    if math.isnan(f_scores):
        f_scores = 0
    return f_scores


def saliency_evaluation(h5_path, method, gt_root, dataset):
    """
    calculate maxFbeta, IoU, and Accuracy for DUTS
    h5_path: the path of the h5 file saving the pseudo-label
    method: the UOD method name
    gt_root: the path saving mask label files
    dataset: dataset name
    """
    # load the h5 file
    file = h5py.File(h5_path, "r")
    method_group = file[method]
    # initialize image names of the dataset
    image_names = list(method_group.keys())
    # initialize result list
    f_scores = []
    accuracies = []
    IoUs = []
    # load each image's annotation
    for image_name in tqdm(image_names):
        # obtain the path of the label file
        mask_path = gt_root + image_name + ".png"
        # load the mask label
        gt_mask = cv2.imread(mask_path)
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        gt_mask[gt_mask > 0] = 1
        # load the prediction
        predicted_masks = method_group[image_name.split('.')[0] + '/masks'][:]

        predicted_masks = predicted_masks.astype(np.uint8)
        if predicted_masks.shape[0] != 1:
            predicted_masks = np.sum(predicted_masks, axis=0)
            predicted_masks[predicted_masks > 0] = 1
        else:
            predicted_masks = predicted_masks[0]
        if predicted_masks.shape != gt_mask.shape:
            predicted_masks = predicted_masks[:gt_mask.shape[0], :gt_mask.shape[1]]
        # calculate each metrics
        f_scores.append(F_max(predicted_masks, gt_mask, beta=0.3))
        accuracies.append(Accuracy(predicted_masks, gt_mask))
        IoUs.append(IoU(predicted_masks, gt_mask))
    # print the result
    print(dataset + " maxFbeta:", sum(f_scores) / len(f_scores))
    print(dataset + " Accuracy:", sum(accuracies) / len(accuracies))
    print(dataset + " IoU:", sum(IoUs) / len(IoUs))
    file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Saliency Evaluation')
    # default arguments
    parser.add_argument('--h5-path', type=str,
                        default="/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/",
                        help='path of the pre-generated h5 file of UOD results')
    parser.add_argument('--uod-method', type=str, default="UnionSeg", choices=["TokenGraphCut", "UnionSeg", "CutLER"],
                        help='UOD method used to generate the h5 file')
    parser.add_argument('--dataset', type=str, default="ECSSD", choices=["ECSSD", "DUTS", "DUTS-OMRON"], help='name of the benchmark')
    parser.add_argument('--gt-path', type=str, default="/home/wzl/DataSet/ECSSD/ground_truth_mask/",
                        help='path of the ground truth folder/file')
    args = parser.parse_args()

    saliency_evaluation(args.h5_path,
                        method=args.uod_method,
                        gt_root=args.gt_path,
                        dataset=args.dataset)