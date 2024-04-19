import numpy as np
import torch
import sys, os
import h5py
import cv2
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


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

def IoUs(predict_masks, gt_masks):
    """
    calculate iou between each predicted mask and every ground truth masks
    predict_masks: (m, w, h) ndarray
    gt_masks: (n, w, h) ndarray
    return: (m, n) ndarray
    """
    # obtain the number of predicted masks
    num_predict_masks = predict_masks.shape[0]
    # obtain the number of ground truth masks
    num_gt_masks = gt_masks.shape[0]
    # flatten each mask
    flat_predict_masks = predict_masks.reshape(num_predict_masks, -1)   # m, w * h
    flat_gt_masks = gt_masks.reshape(num_gt_masks, -1)  # n, w * h
    # calculate intersection matrix
    intersection_matrix = flat_predict_masks @ flat_gt_masks.T  # m, n
    # calculate union matrix
    union_matrix = np.tile(flat_predict_masks.sum(axis=1).reshape(-1, 1), (1, num_gt_masks)) + np.tile(flat_gt_masks.sum(axis=1), (num_predict_masks, 1)) - intersection_matrix # m, n
    # calculate iou matrix
    iou_matrix = intersection_matrix / union_matrix # m, n
    # return the result
    return iou_matrix

def VOC_union_accuracy(h5_path, root_path="/home/wzl/", partition="trainval", KPI="iou", thresh=0.1, method="UnionCut", num=None, instance_nums=None):
    """
    evaluate the accuracy of the UnionCut
    """
    from tqdm import tqdm
    # obtain image names
    if partition == "trainval":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt",
            'r')
    elif partition == "train":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
            'r')
    elif partition == "val":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
            'r')

    # load the h5 file
    file = h5py.File(h5_path, "r")
    # initialize image names of the dataset
    method_group = file[method]
    image_names = list(set([line.strip('\n') for line in fh]) & set(method_group.keys()))
    # obtain semantic label paths
    semantic_label_paths = [
        root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/" + image_name + ".png" for
        image_name in image_names]
    instance_label_paths = [
        root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationObject/" + image_name + ".png"
        for image_name in image_names]
    fh.close()
    # initialize accuracy
    accuracy = 0
    # initialize image num
    image_num = 0
    for i, image_name, semantic_label_path, instance_label_path in tqdm(zip(range(len(image_names)), image_names, semantic_label_paths, instance_label_paths)):
        if num is not None:
            if instance_nums is None:
                # obtain the number of instances in the image
                instance_label = cv2.imread(instance_label_path)
                instance_label = cv2.cvtColor(instance_label, cv2.COLOR_BGR2RGB)
                # change the label to binary mask
                unique_colors = np.unique(instance_label.reshape(-1, 3), axis=0)
                instance_num = len(unique_colors)
                if np.array([0, 0, 0]) in unique_colors:
                    instance_num -= 1
                if np.array([224, 224, 192]) in unique_colors:
                    instance_num -= 1
            else:
                instance_num = instance_nums[i]
            # judge if the image should be counted
            if num == 1:
                if instance_num > num:
                    continue
            elif num == 3:
                if instance_num not in [2, 3]:
                    continue
            elif num == 5:
                if instance_num not in [4, 5]:
                    continue
            else:
                if instance_num < 6:
                    continue

        # load the label
        semantic_label = cv2.imread(semantic_label_path)
        # change the label to binary mask
        semantic_label_mask = np.ones(semantic_label.shape[:-1], np.uint8)
        semantic_label_mask[np.all(semantic_label.reshape((-1, 3)) == np.array([0, 0, 0]), axis=-1).reshape(semantic_label.shape[:-1])] = 0
        # load the union mask of the image
        if method == "UnionCut":
            union_mask = method_group[image_name + '/union'][:]
        else:
            union_mask = np.sum(method_group[image_name + '/masks'][:], axis=0)
            union_mask[union_mask>1] = 1
        # calculate the KPI
        if KPI == "iou":
            kpi = IoU(union_mask, semantic_label_mask)
        elif KPI == "precision":
            kpi = Precision(union_mask, semantic_label_mask)
        elif KPI == "recall":
            kpi = Precision(semantic_label_mask, union_mask)
        elif KPI == "f1":
            precision = Precision(union_mask, semantic_label_mask)
            recall = Precision(semantic_label_mask, union_mask)
            kpi = 2 * precision * recall / (precision + recall)
        # judge if the union mask is good
        if kpi >= thresh:
            accuracy += 1
        image_num += 1
    # close the h5 file
    file.close()
    return accuracy / image_num

def VOC_union_precision(h5_path, root_path="/home/wzl/", partition="trainval", method="UnionCut"):
    """
    evaluate the precision of the UnionCut
    """
    from tqdm import tqdm
    # obtain image names
    if partition == "trainval":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt",
            'r')
    elif partition == "train":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
            'r')
    elif partition == "val":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
            'r')

    # load the h5 file
    file = h5py.File(h5_path, "r")
    # initialize image names of the dataset
    method_group = file[method]
    image_names = list(set([line.strip('\n') for line in fh]) & set(method_group.keys()))
    # obtain semantic label paths
    label_paths = [
        root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/" + image_name + ".png" for
        image_name in image_names]
    fh.close()
    # initialize precision
    precision = 0
    for image_name, semantic_label_path in tqdm(zip(image_names, label_paths)):
        # load the label
        semantic_label = cv2.imread(semantic_label_path)
        # change the label to binary mask
        semantic_label_mask = np.ones(semantic_label.shape[:-1], np.uint8)
        semantic_label_mask[np.all(semantic_label.reshape((-1, 3)) == np.array([0, 0, 0]), axis=-1).reshape(semantic_label.shape[:-1])] = 0
        # load the union mask of the image
        union_mask = method_group[image_name + '/union'][:]
        # calculate the KPI
        precision += Precision(union_mask, semantic_label_mask)
    # close the h5 file
    file.close()
    return precision / len(image_names)

def VOC_union_recall(h5_path, root_path="/home/wzl/", partition="trainval", method="UnionCut"):
    """
    evaluate the recall of the UnionCut
    """
    from tqdm import tqdm
    # obtain image names
    if partition == "trainval":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt",
            'r')
    elif partition == "train":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
            'r')
    elif partition == "val":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
            'r')

    # load the h5 file
    file = h5py.File(h5_path, "r")
    # initialize image names of the dataset
    method_group = file[method]
    image_names = list(set([line.strip('\n') for line in fh]) & set(method_group.keys()))
    # obtain semantic label paths
    label_paths = [
        root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/" + image_name + ".png" for
        image_name in image_names]
    fh.close()
    # initialize recall
    recall = 0
    for image_name, semantic_label_path in tqdm(zip(image_names, label_paths)):
        # load the label
        semantic_label = cv2.imread(semantic_label_path)
        # change the label to binary mask
        semantic_label_mask = np.ones(semantic_label.shape[:-1], np.uint8)
        semantic_label_mask[np.all(semantic_label.reshape((-1, 3)) == np.array([0, 0, 0]), axis=-1).reshape(semantic_label.shape[:-1])] = 0
        # load the union mask of the image
        union_mask = method_group[image_name + '/union'][:]
        # calculate the KPI
        recall += Precision(semantic_label_mask, union_mask)
    # close the h5 file
    file.close()
    return recall / len(image_names)

def VOC_pseudo_mask_mPQ(h5_path, root_path="/home/wzl/", method="TokenCut", partition="trainval"):
    from tqdm import tqdm
    # load the h5 file
    file = h5py.File(h5_path, "r")
    # obtain image names
    if partition == "trainval":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt",
            'r')
    elif partition == "train":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
            'r')
    elif partition == "val":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
            'r')
    # initialize image names of the dataset
    method_group = file[method]
    image_names = list(set([line.strip('\n') for line in fh]) & set(method_group.keys()))
    # obtain semantic label paths
    label_paths = [root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationObject/" + image_name + ".png" for image_name in image_names]
    fh.close()
    # initialize recall
    mPQ = 0
    for image_name, instance_label_path in tqdm(zip(image_names, label_paths)):
        # load the prediction of UOD
        predicted_masks = method_group[image_name + '/masks'][:]
        # load the label
        instance_label = cv2.imread(instance_label_path)
        instance_label = cv2.cvtColor(instance_label, cv2.COLOR_BGR2RGB)
        # change the label to binary mask
        unique_colors = np.unique(instance_label.reshape(-1, 3), axis=0)
        # initialize an array to store instance masks
        instance_label_masks = []
        for color in unique_colors:
            # skip colours of background and border
            if np.array_equal(color, np.array([0, 0, 0])) or np.array_equal(color, np.array([224, 224, 192])):
                continue
            # initialize a mask
            instance_label_mask = np.zeros(instance_label.shape[:-1], np.int_)
            # update label mask
            instance_label_mask[
                np.all(instance_label.reshape((-1, 3)) == color, axis=-1).reshape(
                    instance_label.shape[:-1])] = 1
            instance_label_masks.append(instance_label_mask)
        instance_label_masks = np.stack(instance_label_masks, axis=0)
        # obtain iou matrix between predicted masks and gt masks
        iou_matrix = IoUs(predicted_masks, instance_label_masks)
        # filter TP with threshold 0.5
        iou_matrix = iou_matrix * (iou_matrix > 0.5).astype(np.int_)
        confusion_matrix = (iou_matrix > 0).astype(np.int_)
        # obtain the number of TP
        TP = np.sum(confusion_matrix)
        # obtain the number of FP
        FP = predicted_masks.shape[0] - TP
        # obtain the number of FN
        FN = instance_label_masks.shape[0] - TP
        # calculate PQ
        PQ = np.sum(iou_matrix) / (TP + 0.5 * FP + 0.5 * FN)
        # update mPQ
        mPQ += PQ

    # close the h5 file
    file.close()
    return mPQ / len(image_names)

def VOC_image_instance_num(h5_path, root_path="/home/wzl/", partition="trainval", method="UnionCut"):
    """
    calculate the instance number of each image
    """
    from tqdm import tqdm
    # obtain image names
    if partition == "trainval":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt",
            'r')
    elif partition == "train":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt",
            'r')
    elif partition == "val":
        fh = open(
            root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt",
            'r')

    # load the h5 file
    file = h5py.File(h5_path, "r")
    # initialize image names of the dataset
    method_group = file[method]
    image_names = list(set([line.strip('\n') for line in fh]) & set(method_group.keys()))
    # obtain semantic label paths

    instance_label_paths = [
        root_path + "VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationObject/" + image_name + ".png"
        for image_name in image_names]
    fh.close()
    # initialize image num
    instance_nums = []
    for instance_label_path in tqdm(instance_label_paths):
        # obtain the number of instances in the image
        instance_label = cv2.imread(instance_label_path)
        instance_label = cv2.cvtColor(instance_label, cv2.COLOR_BGR2RGB)
        # change the label to binary mask
        unique_colors = np.unique(instance_label.reshape(-1, 3), axis=0)
        instance_num = len(unique_colors)
        if np.array([0, 0, 0]) in unique_colors:
            instance_num -= 1
        if np.array([224, 224, 192]) in unique_colors:
            instance_num -= 1
        instance_nums.append(instance_num)
    return instance_nums

if __name__ == '__main__':
    # print(DeepLabV3plusEvaluation('/home/wzl/PhysicalImageAugmentation/Evaluation/DeeplabV3plus/module/net_params_' + str(60000) + '.pkl'))
    # print(SOLOv2Evaluation("/home/wzl/PhysicalImageAugmentation/Evaluation/SOLOv2/module/net_params_30000.pkl"))
    # print(voc_optimal_transformation_profit())

    # print(SOLOv2EvaluationPseudo(net_path="/home/wzl/PhysicalImageAugmentation/Evaluation/SOLOv2/module/net_params_TokenCut_5_4_mean_feature_8000_0.02651803787561861.pkl", root_path="/home/wzl/VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/",
    #                                        h5_path="/home/wzl/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5",
    #                                        method="TokenCut", fold=5, fold_num=4,partition="val", cluster=True,
    #                                        n_cluster=1, feature_mode="mean", feature="feature"))

    # # initialize h5 path
    # h5_path = "/home/wzl/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5"
    # # obtain paths of parameter files
    # pkl_names = os.listdir("../SOLOv2/module/")
    # # initialize method list
    # methods = ["MaskCut", "LOST", "MaskDistill", "TokenCut"]
    # # obtain weight pkls for each method
    # net_paths = {}
    # for method in methods:
    #     net_paths[method] = []
    #     for pkl_name in pkl_names:
    #         if method in pkl_name:
    #             net_paths[method].append("../SOLOv2/module/" + pkl_name)
    #     # sort the net path by fold num
    #     net_paths[method].sort(key=lambda x: int(x.split('_')[4]))
    # # evaluation
    # mPQs = {}
    # for method in methods:
    #     mPQ = 0
    #     for i, net_path in zip(range(len(net_paths[method])),net_paths[method]):
    #         mPQ += SOLOv2EvaluationPseudo(net_path=net_path, root_path="/home/wzl/VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/",
    #                                        h5_path=h5_path,
    #                                        method=method, fold=5, fold_num=i,partition="test", cluster=True,
    #                                        n_cluster=30, feature_mode="cls", feature="feature")
    #     mPQs[method] = mPQ / 5
    # print(mPQs)
    # # {'MaskCut': 0.27774986984687844, 'LOST': 0.14895512177852163, 'MaskDistill': 0.24041522327657622, 'TokenCut': 0.28699691120026805}

    # ### Instance ###
    # setup_seed(seed=3407)
    # # initialize h5 path
    # h5_path = "/home/wzl/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5"
    # # h5_path = "/home/wzl/PhysicalImageAugmentation/PseudoMaskGenration/graph_pseudo_labels.h5"
    # # obtain paths of parameter files
    # method = "MaskCut"
    # feature = "feature"
    # feature_mode = "cls"
    # fold_num = 3
    # cls_num = 1
    # net_path = "/home/wzl/PhysicalImageAugmentation/Evaluation/SOLOv2/module/binary/net_params_MaskCut_5_3_1_cls_feature_6000_0.4413492046456698.pkl"
    # # evaluation
    # mPQ = SOLOv2EvaluationPseudo(net_path=net_path,
    #                                       root_path="/home/wzl/VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/",
    #                                       h5_path=h5_path,
    #                                       method=method, fold=5, fold_num=fold_num, partition="test", cluster=True,
    #                                       n_cluster=cls_num, feature_mode="cls", feature=feature)
    # print(mPQ)

    # ### Semantic ###
    # # initialize h5 path
    # # h5_path = "/home/wzl/PhysicalImageAugmentation/PseudoMaskGenration/Baseline_CityScapes_pseudo_labels.h5"
    # h5_path = "/home/wzl/PhysicalImageAugmentation/PseudoMaskGenration/GraphBased_CityScapes_pseudo_labels.h5"
    # # obtain paths of parameter files
    # method = "GraphBased"
    # feature = "feature"
    # feature_mode = "cls"
    # random_seed = 3407
    # setup_seed(seed=random_seed)
    # cls_num = 40
    # net_path = "/home/wzl/PhysicalImageAugmentation/Evaluation/DeeplabV3plus/module/net_params_GraphBased_299_3407_cls_feature_24000_0.13382304935415343.pkl"
    # # evaluation
    # mIoU = DeepLabV3plusEvaluationPseudo(net_path=net_path, root_path="/media/wzl/removable_ssd/Cityscapes/",
    #                                    h5_path=h5_path,
    #                                    method=method, random_seed=random_seed, partition="val", cluster=True,
    #                                    n_cluster=cls_num, feature_mode=feature_mode, feature=feature)
    # print(mIoU)

    # print(VOC_pseudo_mask_mPQ(h5_path="/media/wzl/T7/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5", partition="trainval",method="LOST"))
    # print(VOC_pseudo_mask_mPQ(
    #     h5_path="/media/wzl/T7/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5",
    #     partition="trainval", method="MaskDistill"))
    # print(VOC_pseudo_mask_mPQ(
    #     h5_path="/media/wzl/T7/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5",
    #     partition="trainval", method="TokenCut"))
    # print(VOC_pseudo_mask_mPQ(
    #     h5_path="/media/wzl/T7/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5",
    #     partition="trainval", method="MaskCut"))
    # print(VOC_pseudo_mask_mPQ(
    #     h5_path="/media/wzl/T7/PhysicalImageAugmentation/TokenGraphCutN1_VOC.h5",
    #     partition="trainval", method="TokenGraphCut"))
    # print(VOC_pseudo_mask_mPQ(
    #     h5_path="/media/wzl/T7/PhysicalImageAugmentation/TokenGraphCutN1NoCornerPrior_VOC.h5",
    #     partition="trainval", method="TokenGraphCut"))
    # print(VOC_pseudo_mask_mPQ(
    #     h5_path="/media/wzl/T7/PhysicalImageAugmentation/TokenGraphCutN3_VOC.h5",
    #     partition="trainval", method="TokenGraphCut"))
    # print(VOC_pseudo_mask_mPQ(
    #     h5_path="/media/wzl/T7/PhysicalImageAugmentation/MaskGraphCutN3_VOC.h5",
    #     partition="trainval", method="MaskGraphCut"))
    # print(VOC_pseudo_mask_mPQ(
    #     h5_path="/media/wzl/T7/PhysicalImageAugmentation/MaskGraphCutN1NoCornerPrior_VOC.h5",
    #     partition="trainval", method="MaskGraphCut"))

    h5_path = "/media/wzl/T7/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5"   #"/media/wzl/T7/PhysicalImageAugmentation/MaskGraphCutN3_VOC.h5"    #"/media/wzl/T7/PhysicalImageAugmentation/UnionCut_VOC.h5"
    method = "LOST" #"MaskCut"  #"UnionCut"
    # # obtain instance nums
    # instance_nums = VOC_image_instance_num(h5_path=h5_path, method=method)
    # for i in np.arange(0.1, 1, 0.1):
    #     for j in [1, 3, 5, 7]:
    #         print("iou", i, j, VOC_union_accuracy(h5_path=h5_path, KPI="iou", thresh=i, num=j, method=method, instance_nums=instance_nums))
    #         print("precision", i, j, VOC_union_accuracy(h5_path=h5_path, KPI="precision", thresh=i, num=j, method=method, instance_nums=instance_nums))
    #         print("recall", i, j, VOC_union_accuracy(h5_path=h5_path, KPI="recall", thresh=i, num=j, method=method, instance_nums=instance_nums))
    #         print("f1", i, j, VOC_union_accuracy(h5_path=h5_path, KPI="f1", thresh=i, num=j, method=method, instance_nums=instance_nums))
    #         # print(VOC_union_accuracy(h5_path="/media/wzl/T7/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5", KPI="iou", thresh=i, method="MaskCut"))
    for i in np.arange(0.1, 1, 0.1):
        print("iou", i, VOC_union_accuracy(h5_path=h5_path, KPI="iou", thresh=i, method=method))
        print("precision", i, VOC_union_accuracy(h5_path=h5_path, KPI="precision", thresh=i, method=method))
        print("recall", i, VOC_union_accuracy(h5_path=h5_path, KPI="recall", thresh=i, method=method))
        print("f1", i, VOC_union_accuracy(h5_path=h5_path, KPI="f1", thresh=i, method=method))

    # print(VOC_union_precision(h5_path="/media/wzl/T7/PhysicalImageAugmentation/UnionCut_VOC.h5"))