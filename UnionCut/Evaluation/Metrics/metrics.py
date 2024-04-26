"""
AS for SOLOv2 used for evaluation, please refer to this unofficial implementation:
https://github.com/HibikiJie/SOLOV2/tree/master

Our implemented SOLOv2 was based on that project.
However, since there is no license allocated to that project by the repository owner, we cannot share our customized implementation for now.
"""

import numpy as np
import torch
import sys, os
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

def Pseudo_PQ(net, dataloader, device):
    """
    evaluate the performance of the instance/panoptic segmentation model on the given dataset trained by pseudo labels
    net: model to be evaluated
    dataloader: dataset wrapped in a dataloader
    device: cuda or cpu
    """
    from Evaluation.SOLOv2.predict import predict
    from Evaluation.SOLOv2.DataLoader import get_image_and_annotations_per_batch
    from tqdm import tqdm

    mPQ = 0
    for step, (image_paths, semantic_label_paths, instance_label_paths) in tqdm(enumerate(dataloader)):
        torch_images, categories, torch_instance_label_masks, bboxes = get_image_and_annotations_per_batch(image_paths,
                                                                                                           semantic_label_paths,
                                                                                                           instance_label_paths,
                                                                                                           mass_center=False)
        # move the data to target device
        torch_images = torch_images.to(device)
        # obtain the output of the model
        (confs4, kernels4), (confs8, kernels8), (confs16, kernels16), (confs32, kernels32), (confs64, kernels64), searches = net(torch_images)
        # collect confidences and kernels
        confs = [confs4[0], confs8[0], confs16[0], confs32[0], confs64[0]]
        kernels = [kernels4[0], kernels8[0], kernels16[0], kernels32[0], kernels64[0]]
        try:
            predicted_categories, predicted_masks, predicted_bboxes = predict(confs, kernels, searches)
        except:
            continue
        if predicted_categories is None:
            continue
        else:
            if len(predicted_categories) == 0:
                continue
            # obtain the ground truth masks, bboxes and categories
            gt_masks = torch_instance_label_masks[0].numpy()
            gt_categories = [0 for _ in categories[0]]
            # obtain iou matrix between predicted masks and gt masks
            iou_matrix = IoUs(predicted_masks, gt_masks)
            # only ious that prediction matches gt are counted
            iou_matrix = iou_matrix * (np.tile(predicted_categories.reshape(-1, 1), (1, len(gt_categories))) == np.tile(gt_categories, (len(predicted_categories), 1))).astype(np.int_)
            # filter TP with threshold 0.5
            iou_matrix = iou_matrix * (iou_matrix > 0.5).astype(np.int_)
            confusion_matrix = (iou_matrix > 0).astype(np.int_)
            # obtain the number of TP
            TP = np.sum(confusion_matrix)
            # obtain the number of FP
            FP = len(predicted_categories) - TP
            # obtain the number of FN
            FN = len(gt_categories) - TP
            # calculate PQ
            PQ = np.sum(iou_matrix) / (TP + 0.5 * FP + 0.5 * FN)
            # update mPQ
            mPQ += PQ
    return mPQ / len(dataloader)


def PQ(net, dataloader, device):
    """
    evaluate the performance of the instance/panoptic segmentation model on the given dataset
    """
    from Evaluation.SOLOv2.predict import predict
    from Evaluation.SOLOv2.DataLoader import get_image_and_annotations_per_batch
    from tqdm import tqdm

    mPQ = 0
    for step, (image_paths, semantic_label_paths, instance_label_paths) in tqdm(enumerate(dataloader)):
        torch_images, categories, torch_instance_label_masks, bboxes = get_image_and_annotations_per_batch(image_paths, semantic_label_paths, instance_label_paths, mass_center=False)
        # move the data to target device
        torch_images = torch_images.to(device)
        # obtain the output of the model
        (confs4, kernels4), (confs8, kernels8), (confs16, kernels16), (confs32, kernels32), (confs64, kernels64), searches = net(torch_images)
        # collect confidences and kernels
        confs = [confs4[0], confs8[0], confs16[0], confs32[0], confs64[0]]
        kernels = [kernels4[0], kernels8[0], kernels16[0], kernels32[0], kernels64[0]]
        predicted_categories, predicted_masks, predicted_bboxes = predict(confs, kernels, searches)
        if predicted_categories is None:
            continue
        else:
            # obtain the ground truth masks, bboxes and categories
            gt_masks = torch_instance_label_masks[0].numpy()
            gt_categories = categories[0]
            # obtain iou matrix between predicted masks and gt masks
            iou_matrix = IoUs(predicted_masks, gt_masks)
            # only ious that prediction matches gt are counted
            iou_matrix = iou_matrix * (np.tile(predicted_categories.reshape(-1, 1), (1, len(gt_categories))) == np.tile(gt_categories, (len(predicted_categories), 1))).astype(np.int_)
            # filter TP with threshold 0.5
            iou_matrix = iou_matrix * (iou_matrix > 0.5).astype(np.int_)
            confusion_matrix = (iou_matrix > 0).astype(np.int_)
            # obtain the number of TP
            TP = np.sum(confusion_matrix)
            # obtain the number of FP
            FP = len(predicted_categories) - TP
            # obtain the number of FN
            FN = len(gt_categories) - TP
            # calculate PQ
            PQ = np.sum(iou_matrix) / (TP + 0.5 * FP + 0.5 * FN)
            # update mPQ
            mPQ += PQ
    return mPQ / len(dataloader)

def SOLOv2Evaluation(net_path):
    from Evaluation.SOLOv2.DataLoader import VOC_InstanceSegmentation
    from Evaluation.SOLOv2.net import SOLOv2
    import torch.utils.data as Data

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load the model
    net = SOLOv2(cls_num=20)
    net.load_state_dict(torch.load(net_path))
    net.to(device)
    net.eval()
    # initialize dataset
    testset = VOC_InstanceSegmentation(partition="val")
    # wrap the dataset with DataLoader
    loader = Data.DataLoader(
        dataset=testset,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=1,
    )
    mPQ = PQ(net, loader, device)
    return mPQ

def SOLOv2EvaluationPseudo(model=None, net_path="/home/wzl/PhysicalImageAugmentation/Evaluation/SOLOv2/module/net_params_TokenCut_5_0_cls_feature_21000.pkl", root_path="/db/shared/segmentation/VOCdevkit/",
                                       h5_path="/home/psxzw11/PhysicalImageAugmentation/PseudoMaskGenration/baseline_voc_pseudo_labels.h5",
                                       method="TokenCut", fold=5, fold_num=0,partition="train", cluster=True,
                                       n_cluster=30, feature_mode="cls", feature="feature"):
    from PseudoMaskGeneration.load_h5 import VOC_PseudoLabelLoaderInstance, VOC_PseudoLabelLoader
    from Evaluation.SOLOv2.DataLoader import VOC_InstanceSegmentation
    from Evaluation.SOLOv2.net import SOLOv2
    import torch.utils.data as Data

    # initialize device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if model is None:
        # load the model
        model = SOLOv2(cls_num=n_cluster)
        model.load_state_dict(torch.load(net_path))
        # model = torch.load(net_path)
    model.to(device)
    model.eval()
    # initialize dataset
    testset = VOC_InstanceSegmentation(partition=partition,
                                       image_names=VOC_PseudoLabelLoader(root_path=root_path, h5_path=h5_path, method=method, fold=fold,
                                          fold_num=fold_num, partition=partition, cluster=False, n_cluster=n_cluster,
                                          feature_mode=feature_mode, feature=feature, instance=True).image_names)
    # wrap the dataset with DataLoader
    loader = Data.DataLoader(
        dataset=testset,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        num_workers=1,
    )
    mPQ = Pseudo_PQ(model, loader, device)
    return mPQ

def setup_seed(seed=3407):
    import random
    import os
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    ### Instance ###
    setup_seed(seed=3407)
    # initialize h5 path
    h5_path = "/home/wzl/PhysicalImageAugmentation/PseudoMaskGeneration/baseline_voc_pseudo_labels.h5"
    # h5_path = "/home/wzl/PhysicalImageAugmentation/PseudoMaskGenration/graph_pseudo_labels.h5"
    # obtain paths of parameter files
    method = "MaskCut"
    feature = "feature"
    feature_mode = "cls"
    fold_num = 3
    cls_num = 1
    net_path = "/home/wzl/PhysicalImageAugmentation/Evaluation/SOLOv2/module/binary/net_params_MaskCut_5_3_1_cls_feature_6000_0.4413492046456698.pkl"
    # evaluation
    mPQ = SOLOv2EvaluationPseudo(net_path=net_path,
                                          root_path="/home/wzl/VOC/VOC2012/VOCtrainval_11-May-2012/VOCdevkit/",
                                          h5_path=h5_path,
                                          method=method, fold=5, fold_num=fold_num, partition="test", cluster=True,
                                          n_cluster=cls_num, feature_mode="cls", feature=feature)
    print(mPQ)
