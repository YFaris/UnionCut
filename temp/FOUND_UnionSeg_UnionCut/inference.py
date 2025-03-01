import torch.nn as nn
import torch
from UnionSeg.model import UnionSeg
import numpy as np
import UnionSeg.utils as utils
import UnionSeg.vision_transformer as vits
import cv2
from torchvision import transforms as pth_transforms
import torch.nn.functional as F
from UnionSeg.NCut import ncut
from UnionSeg.NCut import detect_box as tokencut_detect_box
from UnionSeg.crf import densecrf
from sklearn.cluster import DBSCAN
from scipy import ndimage
from collections import Counter
import h5py



def Precision(mask, gt_mask):
    """
    calculate precision between mask1 and mask2
    mask: (w, h) ndarray
    gt_mask: (w, h) ndarray
    """
    intersection = mask * gt_mask
    return 0 if np.sum(mask) == 0 else np.sum(intersection) / np.sum(mask)


def IoU(mask1, mask2):
    """
    calculate iou between mask1 and mask2
    mask1: (w, h) ndarray
    mask2: (w, h) ndarray
    """
    intersection = mask1 * mask2
    union = mask1 + mask2 - intersection
    return 0 if np.sum(union) == 0 else np.sum(intersection) / np.sum(union)


class UnionSegMaster:
    def __init__(self, image, tau=0.2, eps=1e-5, UnionSegPatchSize=8, NCutPatchSize=8, UnionSegResize=224,
                 NCutResize=480, device="cpu", pretrained_dino=None, pretrained_UnionSeg=None, theta=0.5,
                 UnionSegOutput=None, NCutFeatures=None, NCutCosSimilarityMatrix=None, use_cupy=True, N=5, h5_path=None, group_name=None, image_name=None):
        """
        image: image to be segmented
        tau: threshold for attention map construction
        eps: graph edge weight of non-relative links, 1e-5 for TokenCut
        patch_size: size of the patch
        resize_mode: the input size of the image, fixed -> 224x224; flexible - > ratio does not change
        device: cpu or cuda
        pretrained_dino: pretrained dino
        pretrained_UnionSeg: pretrained UnionSeg
        theta: threshold for valid masks
        UnionSegOutput: the output by UnionSeg, if None then it will be calculated
        NCutFeatures: the output by DINO, if None then it will be calculated
        NCutCosSimilarityMatrix: the distance matrix for NCut, if None then it will be calculated
        use_cupy: use cupy to accelerate
        N: iteration times
        h5_path: path of the precomputed fore union file
        group_name: method group name used to decode the h5 file
        image_name: key of the image's record in the h5 file
        """
        # whether used cupy
        self.use_cupy = use_cupy
        # lazy if the output for UnionSeg and Dino are given
        self.UnionSegOutput = UnionSegOutput
        self.NCutCosSimilarityMatrix = NCutCosSimilarityMatrix
        if use_cupy:
            import cupy as cp
            self.NCutCosSimilarityMatrix = cp.asarray(
                NCutCosSimilarityMatrix) if NCutCosSimilarityMatrix is not None else NCutCosSimilarityMatrix
        self.NCutFeatures = NCutFeatures

        self.image = image.copy()  # a copy of the image
        # initialize device
        self.device = torch.device(device)
        # save parameters for NCut
        self.tau = tau
        self.eps = eps
        # save threshold for valid masks
        self.theta = theta
        # obtain the size of the image
        self.row = image.shape[0]
        self.col = image.shape[1]
        # initialize the maximum exploration times per image
        self.N = N
        # initialize the patch size
        self.UnionSegPatchSize = UnionSegPatchSize
        self.NCutPatchSize = NCutPatchSize
        # initialize preprocess mode of resize
        self.UnionSegResize = UnionSegResize
        self.NCutResize = NCutResize
        # load model
        if pretrained_UnionSeg is not None:
            self.UnionSegModel = pretrained_UnionSeg
        elif UnionSegOutput is not None:
            self.UnionSegModel = None
        else:
            self.UnionSegModel = self.load_pretrained_UnionSeg()  # UnionSeg model
        if pretrained_dino is not None:
            self.DinoModel = pretrained_dino
        elif NCutCosSimilarityMatrix is not None:
            self.DinoModel = None
        else:
            self.DinoModel = self.load_pretrained_DINO(NCutPatchSize, scale="base")
        self.sigmoid = nn.Sigmoid()
        # move models to test device
        if self.UnionSegModel is not None and self.DinoModel is not None:
            self.UnionSegModel.to(self.device)
            self.DinoModel.to(self.device)
            self.sigmoid.to(self.device)
        # obtain the size of the feature map
        if self.UnionSegResize is not None:
            self.UnionSeg_feat_row, self.UnionSeg_feat_col = self.UnionSegResize // UnionSegPatchSize, self.UnionSegResize // UnionSegPatchSize
        else:
            self.UnionSeg_feat_row, self.UnionSeg_feat_col = int(np.ceil(self.row / self.UnionSegPatchSize)), int(
                np.ceil(self.col / self.UnionSegPatchSize))
        if self.NCutResize is not None:
            self.NCut_feat_row, self.NCut_feat_col = self.NCutResize // NCutPatchSize, self.NCutResize // NCutPatchSize
        else:
            self.NCut_feat_row, self.NCut_feat_col = int(np.ceil(self.row / NCutPatchSize)), int(
                np.ceil(self.col / NCutPatchSize))

        # h5 file related if precomputed fore union is given
        self.h5_path = None
        self.group_name = None
        self.image_name = None
        if h5_path is not None:
            assert group_name is not None, "The method group saving the fore union should be provided!!!"
            self.h5_path = h5_path
            self.group_name = group_name
            # load the h5 file
            file = h5py.File(h5_path, "r")
            # initialize image names of the dataset
            method_group = file[group_name]
            # check if the image's record exists in the h5 file
            if image_name in list(method_group.keys()):
                self.image_name = image_name
            # close the h5 file
            file.close()


    @staticmethod
    def load_pretrained_DINO(patch_size, scale):
        # load the model
        if patch_size == 8:
            if scale == "small":
                net = vits.vit_small(patch_size=8, num_classes=0)
                # load the model's pre-trained weight
                # net.cuda()
                utils.load_pretrained_weights(net,
                                              "/home/wzl/FOUND/UnionSeg/dino_deitsmall8_pretrain.pth",
                                              checkpoint_key=None,
                                              model_name="vit_small",
                                              patch_size=8)
            elif scale == "base":
                net = vits.vit_base(patch_size=8, num_classes=0)
                # load the model's pre-trained weight
                # net.cuda()
                utils.load_pretrained_weights(net,
                                              "/home/wzl/FOUND/UnionSeg/dino_vitbase8_pretrain.pth",
                                              checkpoint_key=None,
                                              model_name="vit_base",
                                              patch_size=8)
        elif patch_size == 16:
            net = vits.vit_small(patch_size=16, num_classes=0)
            # load the model's pre-trained weight
            # net.cuda()
            utils.load_pretrained_weights(net,
                                          "/home/wzl/FOUND/UnionSeg/dino_deitsmall16_pretrain.pth",
                                          checkpoint_key=None,
                                          model_name="vit_small",
                                          patch_size=16)
        net.eval()
        return net

    @staticmethod
    def _preprocess(image, patch_size, resize, feat_row, feat_col):
        """
        preprocess the cv2 image
        mode: fixed -> 224 * 224; flexible -> ratio does not change
        """
        # transfer the image from bgr to rgb
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if resize is not None:
            # resize the image to 224x224
            RGB_image = cv2.resize(RGB_image, (resize, resize), cv2.INTER_LANCZOS4)
        else:
            zeros_array = np.zeros((feat_row * patch_size, feat_col * patch_size, 3)).astype(np.uint8)
            zeros_array[:RGB_image.shape[0], :RGB_image.shape[1], :] = RGB_image.copy()
            RGB_image = zeros_array
        # pytorch preprocessing
        transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        torch_image = transform(RGB_image)
        return torch_image.unsqueeze(0)  # 1, 3, 224, 224

    def _DINO_features(self):
        """
        return features of all patches at the last ViT block
        net: the model
        image: the BGR cv2 image, unnormalized
        """
        # preprocess the image
        torch_image = self._preprocess(image=self.image,
                                       patch_size=self.NCutPatchSize,
                                       resize=self.NCutResize,
                                       feat_row=self.NCut_feat_row,
                                       feat_col=self.NCut_feat_col)
        torch_image = torch_image.to(self.device)
        # obtain the output embeddings of DINO
        with torch.no_grad():
            features, qs, ks, vs = self.DinoModel.get_last_qkv(torch_image)
        return features.squeeze(0).detach().cpu(), \
            qs.permute(0, 2, 1, 3).reshape(1, self.NCut_feat_row * self.NCut_feat_col + 1, -1).squeeze(
                0).detach().cpu(), \
            ks.permute(0, 2, 1, 3).reshape(1, self.NCut_feat_row * self.NCut_feat_col + 1, -1).squeeze(
                0).detach().cpu(), \
            vs.permute(0, 2, 1, 3).reshape(1, self.NCut_feat_row * self.NCut_feat_col + 1, -1).squeeze(0).detach().cpu()

    @staticmethod
    def load_pretrained_UnionSeg():
        """
        load pretrained UnionSeg model
        path: the path of the model
        """
        UnionSegModel = UnionSeg()
        UnionSegModel.decoder_load_weights(
            "/home/wzl/FOUND/UnionSeg/module/decoder_weights_niter600.pt")
        UnionSegModel.eval()
        return UnionSegModel

    @staticmethod
    def _square_distance_matrix(features, distance_mode, feat_nums, dim=768):
        """
        calculate squared distance matrix of each point
        params:
            features: (self.feat_row * self.feat_col, dim)
            dim: 384 for vit small, 768 for vit base
        """
        # resize features to -1, 384
        flatten_features = features.reshape(-1, dim)
        # obtain correlation
        correlation = flatten_features @ flatten_features.T
        if distance_mode == "Euclidean":
            # obtain length square of each feature
            features_square = np.sum(flatten_features ** 2, axis=1)
            # expand the feature squares into a square matrix by repeating
            features_square = np.repeat(features_square, feat_nums, axis=0).reshape(-1, feat_nums)
            # calculate the distance matrix
            square_distance_matrix = features_square - 2 * correlation + features_square.T
            np.fill_diagonal(square_distance_matrix, 0)
            square_distance_matrix[square_distance_matrix < 0] = 0
            return square_distance_matrix, correlation
        elif distance_mode == "Mahalanobis":
            # obtain x - y between each points
            differences = (flatten_features.reshape(feat_nums, 1, dim) - flatten_features.reshape(1, feat_nums,
                                                                                                  dim)).reshape(-1, dim)
            # calculate covariance of features
            COV = np.cov(flatten_features, rowvar=False)
            # calculated squared Mahalanobis distance
            squared_Mahalanobis_matrix = np.sum((differences @ np.linalg.inv(COV)) * differences, axis=1).reshape(
                feat_nums,
                feat_nums)
            return squared_Mahalanobis_matrix, correlation

    @staticmethod
    def _norm_features(features):
        """
        normalize features
        """
        return F.normalize(features, p=2).numpy()

    @staticmethod
    def _absolute2relative(indices, feat_col):
        """
        transfer indices of array to matrix
        indices: np.array([i,....])
        """
        j_s = indices % feat_col
        i_s = (indices - j_s) // feat_col
        return np.stack((i_s, j_s)).T

    @staticmethod
    def _check_num_fg_corners(mask):
        # check number of corners belonging to the foreground
        top_l, top_r, bottom_l, bottom_r = mask[0][0], mask[0][-1], mask[-1][0], mask[-1][
            -1]
        nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
        return nc

    def _disconnection_detection(self, mask, crf=True, resize=True):
        """
        split disconnected masks in the given mask
        """
        # initialize a list for saving all isolated masks
        isolated_masks = []
        if resize:
            # resize the mask
            mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_NEAREST)
        # connectivity detection
        num_labels, labels = cv2.connectedComponents(mask)
        for j in range(1, num_labels):
            # obtain an isolate mask
            if resize:
                isolate_mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)
            else:
                isolate_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            isolate_mask[labels == j] = 1
            if crf:
                # apply crf
                isolate_mask = densecrf(self.image, isolate_mask)
            # collect the mask
            if np.sum(isolate_mask) != 0:
                isolated_masks.append(isolate_mask)
        isolated_masks.sort(key=lambda x: np.sum(x), reverse=True)
        return isolated_masks

    @staticmethod
    def _visualization(masks):
        """
        visualize each component with different color
        :return:
        """
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

    @staticmethod
    def _check_previous_error(mask, prev_errors, thresh=0.75):
        """
        check if previous errors happen again
        """
        if len(prev_errors) == 0:
            return False
        vote = 0
        for prev_error in prev_errors:
            if Precision(prev_error, mask) > thresh:
                vote += 1
        return True if vote / len(prev_errors) >= 0.5 else False

    def _TokenCut(self, features, fore_union, left_indices, mode="zero"):
        # initialize wait list
        waitlist = []
        if mode == "remove":
            features = features[left_indices, :]
            # calculate distance matrix
            _, cos_similarity_matrix = self._square_distance_matrix(self._norm_features(features),
                                                                    distance_mode="Euclidean",
                                                                    feat_nums=features.shape[0], dim=features.shape[1])
            # TokenCut
            for _ in range(self.N):
                # NCut
                try:
                    bipartition, seed, second_smallest_vec = ncut(cos_similarity_matrix, self.tau, self.eps, feat_row=self.NCut_feat_row, feat_col=self.NCut_feat_col, left_indices=left_indices)
                    # obtain the seed for bipartition
                    seed_neg = np.argmin(second_smallest_vec)
                    seed_pos = np.argmax(second_smallest_vec)
                    if bipartition[seed] != 1:
                        # reverse the mask
                        bipartition = ~bipartition
                        # change the seed
                        seed = seed_neg if bipartition[seed_neg] else seed_pos
                    # extract the principal component
                    bipartition = bipartition.reshape([self.NCut_feat_row, self.NCut_feat_col]).astype(float)
                    # predict BBox
                    pred, _, objects, cc = tokencut_detect_box(bipartition, seed,
                                                               [self.NCut_feat_row, self.NCut_feat_col],
                                                               scales=[self.NCutPatchSize, self.NCutPatchSize],
                                                               initial_im_size=[
                                                                   self.NCut_feat_row * self.NCutPatchSize,
                                                                   self.NCut_feat_col * self.NCutPatchSize])
                    mask = np.zeros([self.NCut_feat_row, self.NCut_feat_col])
                    mask[cc[0], cc[1]] = 1
                    bipartition = (mask == 1).reshape(-1)
                except:
                    break

                # make the previous detection failure as background
                bipartition[~left_indices] = False
                # obtain the abs indices the subsub-region
                sub_fore_indices, sub_back_indices = np.arange(self.NCut_feat_row * self.NCut_feat_col)[bipartition], \
                    np.arange(self.NCut_feat_row * self.NCut_feat_col)[~bipartition]
                # transfer abs indices to relative indices
                sub_fore_relative_indices = self._absolute2relative(sub_fore_indices, feat_col=self.NCut_feat_col)
                sub_back_relative_indices = self._absolute2relative(sub_back_indices, feat_col=self.NCut_feat_col)
                # obtain mask
                sub_fore_mask, sub_back_mask = np.zeros((self.NCut_feat_row, self.NCut_feat_col),
                                                        dtype=np.uint8), np.zeros(
                    (self.NCut_feat_row, self.NCut_feat_col), dtype=np.uint8)
                sub_fore_mask[sub_fore_relative_indices[:, 0], sub_fore_relative_indices[:, 1]] = 1
                sub_back_mask[sub_back_relative_indices[:, 0], sub_back_relative_indices[:, 1]] = 1
                # judge if the mask is the fore mask
                if len(waitlist) == 0:
                    # this is the first mask
                    if self._check_num_fg_corners(cv2.erode(densecrf(self.image, sub_fore_mask),
                                                            kernel=cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                                             (5, 5)))) >= 3:
                        sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
                    elif Precision(sub_fore_mask, fore_union) < Precision(sub_back_mask, fore_union):
                        sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
                        # self._check_num_fg_corners(sub_fore_mask) >= 3 or
                        if self._check_num_fg_corners(
                                cv2.erode(densecrf(self.image, sub_fore_mask),
                                          kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))) >= 3:
                            sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
                else:
                    if IoU(sub_fore_mask, waitlist[-1]) > 0.5 or Precision(sub_fore_mask, fore_union) < self.theta:
                        break
                    # remove previous mask from the current mask
                    for mask in waitlist:
                        if Precision(mask, sub_fore_mask) > 0.5:
                            sub_fore_mask = sub_fore_mask - mask
                            sub_fore_mask[sub_fore_mask < 0] = 0
                # save the mask
                waitlist.append(sub_fore_mask)
                # mask out the mask
                cos_similarity_matrix[np.arange(features.shape[0])[bipartition[left_indices]], :] = self.eps
                cos_similarity_matrix[:, np.arange(features.shape[0])[bipartition[left_indices]]] = self.eps
                # judge if the bipartition should be finished
                # calculate the overall segmented mask
                overall_segmented_mask = np.sum(np.stack(waitlist), axis=0)
                overall_segmented_mask[overall_segmented_mask > 1] = 1
                if Precision(fore_union, overall_segmented_mask) >= 0.8:
                    # most of the foreground has been segmented
                    break
        elif mode == "zero":
            # calculate distance matrix
            _, cos_similarity_matrix = self._square_distance_matrix(self._norm_features(features),
                                                                    distance_mode="Euclidean",
                                                                    feat_nums=features.shape[0], dim=features.shape[1])
            # mask out the mask
            cos_similarity_matrix[np.arange(features.shape[0])[~left_indices], :] = self.eps
            cos_similarity_matrix[:, np.arange(features.shape[0])[~left_indices]] = self.eps
            # TokenCut
            for _ in range(self.N):
                # NCut
                try:
                    bipartition, seed, second_smallest_vec = ncut(cos_similarity_matrix, self.tau, self.eps, feat_row=self.NCut_feat_row, feat_col=self.NCut_feat_col)
                    # obtain the seed for bipartition
                    seed_neg = np.argmin(second_smallest_vec)
                    seed_pos = np.argmax(second_smallest_vec)
                    if bipartition[seed] != 1:
                        # reverse the mask
                        bipartition = ~bipartition
                        # change the seed
                        seed = seed_neg if bipartition[seed_neg] else seed_pos
                    # extract the principal component
                    bipartition = bipartition.reshape([self.NCut_feat_row, self.NCut_feat_col]).astype(float)
                    # predict BBox
                    pred, _, objects, cc = tokencut_detect_box(bipartition, seed,
                                                               [self.NCut_feat_row, self.NCut_feat_col],
                                                               scales=[self.NCutPatchSize, self.NCutPatchSize],
                                                               initial_im_size=[
                                                                   self.NCut_feat_row * self.NCutPatchSize,
                                                                   self.NCut_feat_col * self.NCutPatchSize])
                    mask = np.zeros([self.NCut_feat_row, self.NCut_feat_col])
                    mask[cc[0], cc[1]] = 1
                    bipartition = (mask == 1).reshape(-1)
                except:
                    break
                # make the previous detection failure as background
                bipartition[~left_indices] = False
                # obtain the abs indices the subsub-region
                sub_fore_indices, sub_back_indices = np.arange(self.NCut_feat_row * self.NCut_feat_col)[bipartition], \
                    np.arange(self.NCut_feat_row * self.NCut_feat_col)[~bipartition]
                # transfer abs indices to relative indices
                sub_fore_relative_indices = self._absolute2relative(sub_fore_indices, feat_col=self.NCut_feat_col)
                sub_back_relative_indices = self._absolute2relative(sub_back_indices, feat_col=self.NCut_feat_col)
                # obtain mask
                sub_fore_mask, sub_back_mask = np.zeros((self.NCut_feat_row, self.NCut_feat_col),
                                                        dtype=np.uint8), np.zeros(
                    (self.NCut_feat_row, self.NCut_feat_col), dtype=np.uint8)
                sub_fore_mask[sub_fore_relative_indices[:, 0], sub_fore_relative_indices[:, 1]] = 1
                sub_back_mask[sub_back_relative_indices[:, 0], sub_back_relative_indices[:, 1]] = 1
                # judge if the mask is the fore mask
                if len(waitlist) == 0:
                    # this is the first mask
                    if self._check_num_fg_corners(cv2.erode(densecrf(self.image, sub_fore_mask),
                                                            kernel=cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                                             (5, 5)))) >= 3:
                        sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
                        sub_fore_indices, sub_back_indices = sub_back_indices, sub_fore_indices
                    elif Precision(sub_fore_mask, fore_union) < Precision(sub_back_mask, fore_union):
                        sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
                        sub_fore_indices, sub_back_indices = sub_back_indices, sub_fore_indices
                        # self._check_num_fg_corners(sub_fore_mask) >= 3 or
                        if self._check_num_fg_corners(
                                cv2.erode(densecrf(self.image, sub_fore_mask),
                                          kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))) >= 3:
                            sub_fore_mask, sub_back_mask = sub_back_mask, sub_fore_mask
                            sub_fore_indices, sub_back_indices = sub_back_indices, sub_fore_indices
                else:
                    if IoU(sub_fore_mask, waitlist[-1]) > 0.5 or Precision(sub_fore_mask, fore_union) < self.theta:
                        break
                    # remove previous mask from the current mask
                    for mask in waitlist:
                        if Precision(mask, sub_fore_mask) > 0.5:
                            sub_fore_mask = sub_fore_mask - mask
                            sub_fore_mask[sub_fore_mask < 0] = 0
                # save the mask
                waitlist.append(sub_fore_mask)
                # mask out the mask
                cos_similarity_matrix[np.arange(features.shape[0])[bipartition], :] = self.eps
                cos_similarity_matrix[:, np.arange(features.shape[0])[bipartition]] = self.eps
                # judge if the bipartition should be finished
                # calculate the overall segmented mask
                overall_segmented_mask = np.sum(np.stack(waitlist), axis=0)
                overall_segmented_mask[overall_segmented_mask > 1] = 1
                if Precision(fore_union, overall_segmented_mask) >= 0.8:
                    # most of the foreground has been segmented
                    break
        return waitlist

    def _MaskCut(self, features, fore_union, zero_indices, removed_indices):
        # initialize wait list
        waitlist = []
        # calculate distance matrix
        if self.NCutCosSimilarityMatrix is None:
            _, cos_similarity_matrix = self._square_distance_matrix(self._norm_features(features),
                                                                    distance_mode="Euclidean",
                                                                    feat_nums=features.shape[0])
        else:
            cos_similarity_matrix = self.NCutCosSimilarityMatrix
        # mask out zero or removed tokens
        zero_indices = zero_indices[~removed_indices]
        cos_similarity_matrix = cos_similarity_matrix[~removed_indices, :]
        cos_similarity_matrix = cos_similarity_matrix[:, ~removed_indices]
        cos_similarity_matrix[zero_indices, :] = 0
        cos_similarity_matrix[:, zero_indices] = 0

        # TokenCut
        for _ in range(self.N):
            # NCut
            try:
                bipartition, seed, second_smallest_vec = ncut(cos_similarity_matrix, self.tau, self.eps,
                                                              feat_row=self.NCut_feat_row, feat_col=self.NCut_feat_col,
                                                              left_indices=~removed_indices, use_cupy=self.use_cupy)
            except:
                try:
                    if self.use_cupy:
                        cos_similarity_matrix = cos_similarity_matrix.get()
                    bipartition, seed, second_smallest_vec = ncut(cos_similarity_matrix, self.tau, self.eps,
                                                                  feat_row=self.NCut_feat_row,
                                                                  feat_col=self.NCut_feat_col,
                                                                  left_indices=~removed_indices, use_cupy=False)
                except:
                    break
            # obtain the seed for bipartition
            seed_neg = np.argmin(second_smallest_vec)
            seed_pos = np.argmax(second_smallest_vec)
            # make the removed previous detection failure as background
            bipartition[removed_indices] = False
            # obtain the abs indices the subsub-region
            sub_fore_indices, sub_back_indices = np.arange(self.NCut_feat_row * self.NCut_feat_col)[bipartition], \
                np.arange(self.NCut_feat_row * self.NCut_feat_col)[~bipartition]
            # transfer abs indices to relative indices
            sub_fore_relative_indices = self._absolute2relative(sub_fore_indices, feat_col=self.NCut_feat_col)
            sub_back_relative_indices = self._absolute2relative(sub_back_indices, feat_col=self.NCut_feat_col)
            # obtain mask
            sub_fore_mask, sub_back_mask = np.zeros((self.NCut_feat_row, self.NCut_feat_col),
                                                    dtype=np.uint8), np.zeros(
                (self.NCut_feat_row, self.NCut_feat_col), dtype=np.uint8)
            sub_fore_mask[sub_fore_relative_indices[:, 0], sub_fore_relative_indices[:, 1]] = 1
            sub_back_mask[sub_back_relative_indices[:, 0], sub_back_relative_indices[:, 1]] = 1
            # judge if the mask is the fore mask
            if self._check_num_fg_corners(sub_fore_mask) >= 3:
                reverse = True
            else:
                reverse = bipartition[seed] != 1
            if reverse:
                # reverse the mask
                bipartition = ~bipartition
                # make the removed previous detection failure as background
                bipartition[removed_indices] = False
                # change the seed
                seed = seed_neg if bipartition[seed_neg] else seed_pos

            # extract the principal component
            bipartition = bipartition.reshape([self.NCut_feat_row, self.NCut_feat_col]).astype(float)
            # predict BBox
            pred, _, objects, cc = tokencut_detect_box(bipartition, seed,
                                                       [self.NCut_feat_row, self.NCut_feat_col],
                                                       scales=[self.NCutPatchSize, self.NCutPatchSize],
                                                       initial_im_size=[self.NCut_feat_row * self.NCutPatchSize,
                                                                        self.NCut_feat_col * self.NCutPatchSize])
            mask = np.zeros([self.NCut_feat_row, self.NCut_feat_col])
            mask[cc[0], cc[1]] = 1
            sub_fore_mask = mask
            bipartition = (sub_fore_mask == 1).reshape(-1)
            # remove previous mask from the current mask
            for mask in waitlist:
                if Precision(mask, sub_fore_mask) > 0.5:
                    sub_fore_mask = sub_fore_mask - mask
                    sub_fore_mask[sub_fore_mask < 0] = 0
                    bipartition = (sub_fore_mask == 1).reshape(-1)
            # save the mask
            waitlist.append(sub_fore_mask)
            # mask out the mask
            cos_similarity_matrix[np.arange(cos_similarity_matrix.shape[0])[bipartition[~removed_indices]], :] = 0
            cos_similarity_matrix[:, np.arange(cos_similarity_matrix.shape[0])[bipartition[~removed_indices]]] = 0
            # judge if the bipartition should be finished
            # calculate the overall segmented mask
            overall_segmented_mask = np.sum(np.stack(waitlist), axis=0)
            overall_segmented_mask[overall_segmented_mask > 1] = 1
            if Precision(fore_union, overall_segmented_mask) >= 0.8:
                # most of the foreground has been segmented
                break
        return waitlist

    def returnUnion(self, vis=False, resize=False, crf=False):
        """
        return the foreground union segmented by UnionSeg
        """
        if self.image_name is not None:
            # the precomputed fore union of the image exists
            # load the h5 file
            file = h5py.File(self.h5_path, "r")
            method_group = file[self.group_name]
            fore_union = method_group[self.image_name + '/union'][:].astype(np.uint8)
            fore_union = cv2.resize(fore_union, (28, 28), cv2.INTER_NEAREST)
            # close the h5 file
            file.close()
        else:
            # preprocess the image
            torch_image = self._preprocess(image=self.image,
                                           patch_size=self.UnionSegPatchSize,
                                           resize=self.UnionSegResize,
                                           feat_row=self.UnionSeg_feat_row,
                                           feat_col=self.UnionSeg_feat_col)
            # move the image to device
            torch_image = torch_image.to(self.device)
            # obtain the prediction of UnionSeg
            preds, _, _, _ = self.UnionSegModel(torch_image)
            # Compute mask detection
            pred_mask = (self.sigmoid(preds.detach()) > 0.5).float()
            # decode the foreground union
            fore_union = pred_mask[0, 0].cpu().detach().numpy().astype(np.uint8)
        # visualization
        if vis:
            cv2.imshow('image', self.image)
            cv2.imshow("fore total",
                       255 * cv2.resize(fore_union, (self.image.shape[1], self.image.shape[0]), cv2.INTER_NEAREST))
            cv2.waitKey(0)
        if resize:
            if crf:
                fore_union = densecrf(self.image, fore_union)
            else:
                fore_union = cv2.resize(fore_union, (self.image.shape[1], self.image.shape[0]), cv2.INTER_NEAREST)
        elif crf:
            fore_union = densecrf(self.image, fore_union)
        return fore_union

    def tokencut_segmentation(self, small_thresh=0.01, vis=False):
        """
        TokenCut + UnionSeg segmentation
        """
        # obtain the foreground Union
        raw_fore_union = self.returnUnion()
        fore_union = cv2.resize(raw_fore_union, (self.NCut_feat_col, self.NCut_feat_row), cv2.INTER_NEAREST)

        # obtain features of the image for NCut
        _, _, ks, _ = self._DINO_features()
        ks = ks[1:, :]
        # initialize indices of unremoved patches
        left_indices = np.ones(ks.shape[0], dtype=np.bool_)
        # save previous errors
        prev_errors = []
        try_mode = "zero"   # make the false area zero or "remove the false error"
        for t in range(5):
            # hierarchical NCut
            waitlist = self._TokenCut(ks, fore_union, left_indices, mode=try_mode)
            # initialize result list
            masks = []
            # remove false positive
            for mask in waitlist:
                # only keep the mask covered by foreground union
                if Precision(mask, fore_union) > self.theta and not self._check_previous_error(mask, prev_errors):
                    masks.append(mask)
            if len(masks) == 0:
                # no satisfied masks discovered
                waitlist.sort(key=lambda x: Precision(x, fore_union))
                # remove the content of the first mask in the wait list from the image
                left_indices[((waitlist[0]) == 1).reshape(-1)] = False
                # save the error
                if t == 0:
                    prev_errors.extend(waitlist)
                # check if the error happens again, then make the mode "remove"
                if t == 1:
                    for mask in waitlist:
                        if self._check_previous_error(mask, prev_errors):
                            try_mode = "remove"
                            print(try_mode)
                            break
            else:
                break
        # if no satisfied masks, make the foreground union given by UnionSeg as the final output
        if len(masks) == 0:
            masks.append(fore_union)
        # apply crf
        crf_masks = []
        for mask in masks:
            crf_masks.extend(self._disconnection_detection(mask))
        # sort the masks by area in descending order
        crf_masks.sort(key=lambda x: np.sum(x), reverse=True)
        masks = np.stack(crf_masks)
        # judge if the fine segmentation is robust
        # obtain the total area of overall mask
        fore_total = np.sum(masks, axis=0)
        fore_total[fore_total > 1] = 1
        fore_total = np.sum(fore_total)
        # remove too small masks
        masks = masks[np.sum(masks, axis=(1, 2)) / fore_total > small_thresh]
        # make sure one mask is returned
        if len(masks) == 0:
            masks = np.stack(self._disconnection_detection(crf_masks[0]))
        if vis:
            cv2.imshow('image', self.image)
            cv2.imshow("fore total", 255 * fore_union)
            # print(len(background_ids), max_thresh)
            # plt.show()
            colorful_masks = self._visualization(masks)
            cv2.imshow("masks", colorful_masks)
            # waiting for user interaction
            pressedKey = cv2.waitKey(0) & 0xFF
            if pressedKey == ord('i'):
                # save the independent case
                cv2.imwrite("/media/wzl/T7/PhysicalImageAugmentation/InstanceCut/DataSet/independent/image/" + image_path.split("/")[-1], image)
                cv2.imwrite("/media/wzl/T7/PhysicalImageAugmentation/InstanceCut/DataSet/independent/mask/" + image_path.split("/")[-1].split(".")[0] + "_mask.png", 255 * masks[0])
            elif pressedKey == ord('o'):
                # save the overlapped case
                cv2.imwrite("/media/wzl/T7/PhysicalImageAugmentation/InstanceCut/DataSet/overlap/image/" +
                            image_path.split("/")[-1], image)
                cv2.imwrite("/media/wzl/T7/PhysicalImageAugmentation/InstanceCut/DataSet/overlap/mask/" +
                            image_path.split("/")[-1].split(".")[0] + "_mask.png", 255 * masks[0])
            elif pressedKey == ord('h'):
                # save the overlapped case
                cv2.imwrite("/media/wzl/T7/PhysicalImageAugmentation/InstanceCut/DataSet/hard/image/" +
                            image_path.split("/")[-1], image)
                cv2.imwrite("/media/wzl/T7/PhysicalImageAugmentation/InstanceCut/DataSet/hard/mask/" +
                            image_path.split("/")[-1].split(".")[0] + "_mask.png", 255 * masks[0])
        return masks, fore_union

    def maskcut_segmentation(self, vis=False):
        """
        MaskCut + UnionSeg segmentation
        """
        # resize the image to fixed size for crf
        I_new = cv2.resize(self.image, (self.NCutResize, self.NCutResize), cv2.INTER_LANCZOS4)

        # check if UnionSeg output is given
        if self.UnionSegOutput is not None:
            raw_fore_union = self.UnionSegOutput
        else:
            # obtain the foreground Union
            raw_fore_union = self.returnUnion()

        fore_union = cv2.resize(raw_fore_union, (self.NCut_feat_col, self.NCut_feat_row), cv2.INTER_NEAREST)

        # fore_union = cv2.resize(raw_fore_union, (self.NCutResize, self.NCutResize), cv2.INTER_NEAREST)
        # fore_union = cv2.resize(densecrf(I_new, fore_union), (self.NCutResize, self.NCutResize), cv2.INTER_NEAREST)
        # fore_union = cv2.resize(fore_union, (self.NCut_feat_col, self.NCut_feat_row), cv2.INTER_NEAREST)

        # check if DINO output is given
        if self.NCutCosSimilarityMatrix is not None:
            ks = None
        else:
            # obtain features of the image for NCut
            _, _, ks, _ = self._DINO_features()
            ks = ks[1:, :]
        # initialize indices of removed patches
        zero_indices = ~np.ones(self.NCut_feat_col * self.NCut_feat_row, dtype=np.bool_)
        removed_indices = ~np.ones(self.NCut_feat_col * self.NCut_feat_row, dtype=np.bool_)
        zero_regions = []
        removed_regions = []
        # save previous errors
        prev_errors = []
        try_mode = "zero"  # make the false area zero or "remove the false error"
        # initialize result list
        masks = []
        for t in range(3):
            # hierarchical NCut
            waitlist = self._MaskCut(ks, fore_union, zero_indices, removed_indices)
            # remove empty mask
            waitlist = [mask for mask in waitlist if np.sum(mask) != 0]
            # remove false positive
            for mask in waitlist:
                # only keep the mask covered by foreground union
                if Precision(mask, fore_union) > self.theta and not self._check_previous_error(mask, prev_errors):
                    masks.append(mask)
                else:
                    print("remove because precision: ", Precision(mask, fore_union))
            if len(masks) == 0:
                # no satisfied masks discovered
                waitlist.sort(key=lambda x: Precision(x, fore_union))
                # mask out try failure to avoid it happens again
                if len(zero_regions) == 0:
                    # zero the content of the error in the wait list from the image
                    zero_regions.append(waitlist[0])
                else:
                    # zero the previous error is not enough, it should be removed
                    removed_regions.append(zero_regions.pop())
                # update indices for zero area and removed area
                zero_indices = np.sum(np.stack(zero_regions), axis=0).reshape(-1) > 0 if len(
                    zero_regions) > 0 else ~np.ones(self.NCut_feat_col * self.NCut_feat_row, dtype=np.bool_)
                removed_indices = np.sum(np.stack(removed_regions), axis=0).reshape(-1) > 0 if len(
                    removed_regions) > 0 else ~np.ones(self.NCut_feat_col * self.NCut_feat_row, dtype=np.bool_)
            else:
                break
        # if no satisfied masks, make the foreground union given by UnionSeg as the final output
        if len(masks) == 0:
            masks.append(fore_union.astype(np.uint8))
        # apply crf
        pseudo_masks = []
        for mask in masks:
            mask = cv2.resize(mask, (self.NCutResize, self.NCutResize), cv2.INTER_NEAREST)
            pseudo_mask = densecrf(I_new, mask)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5)
            # # filter out the mask that have a very different pseudo-mask after the CRF
            # if IoU(mask, pseudo_mask) < 0.5:
            #     pseudo_mask = pseudo_mask * -1
            #     pseudo_mask[pseudo_mask < 0] = 0
            pseudo_masks.append(pseudo_mask)
        # sort the masks by area in descending order
        pseudo_masks.sort(key=lambda x: np.sum(x), reverse=True)
        pseudo_masks = np.stack(pseudo_masks)
        if vis:
            cv2.imshow('image', self.image)
            cv2.imshow("fore total", 255 * fore_union)
            # print(len(background_ids), max_thresh)
            # plt.show()
            colorful_masks = self._visualization(pseudo_masks)
            cv2.imshow("masks",
                       cv2.resize(colorful_masks, (self.image.shape[1], self.image.shape[0]), cv2.INTER_NEAREST))
            cv2.waitKey(0)
        return pseudo_masks, fore_union

    @staticmethod
    def find_bbox(binary_image):
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
        # return normal bbox
        return [xmin, ymin, xmax, ymax]


if __name__ == "__main__":
    import os

    # initialize model
    UnionSegInstance = UnionSegMaster.load_pretrained_UnionSeg()
    DinoInstance = UnionSegMaster.load_pretrained_DINO(patch_size=8, scale="base")

    # # obtain paths of each image
    # root_path = "/home/wzl/DataSet/"
    #
    # # save names of each image
    # image_names = [image.split('.')[0] for image in os.listdir(root_path + "ECSSD/images/")]
    #
    # # save absolute paths of each image in the computer
    # local_paths = ["/home/wzl/DataSet/ECSSD/images/0115.jpg"] + ["/home/wzl/DataSet/ECSSD/images/0371.jpg"] + [root_path + "ECSSD/images/0029.jpg"] + [root_path + "ECSSD/images/" + image_name + ".jpg"
    #                for image_name in image_names]

    # # load image paths
    # local_paths = [
    #     "/home/wzl/DataSet/DUT-OMRON/DUT-OMRON-image/DUT-OMRON-image/" + filename for filename in
    #     os.listdir("/home/wzl/DataSet/DUT-OMRON/DUT-OMRON-image/DUT-OMRON-image/")]

    local_paths = [
        "/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/" + filename for filename in
        os.listdir("/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/")]

    # local_paths = ['009687', '005348', '003817', '009881', '007388', '005542', '009499', '007979', '009738', '006847',
    #                '003580', '005478', '008644', '009627', '002126', '006135', '002415', '005045', '009587', '001758',
    #                '005231', '001061', '009712', '000285', '001727', '000859', '004797', '002947', '007590', '008720',
    #                '001807', '006474', '006180', '000482', '001936', '005510', '009935', '001101', '000259', '006427']
    # local_paths = ["/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/" + filename + ".jpg" for filename in
    #                local_paths]

    # # obtain paths of each image
    # root_path = "/home/wzl/DataSet/imagenet/train/"
    # patch_size = 8
    # resize_mode = "fixed_480"
    # UnionCut_patch_size = 8
    # UnionCut_resize_mode = "fixed_480"
    #
    # # folder names of each image
    # folder_names = os.listdir(root_path)
    # image_names = []
    # local_paths = []
    # image_paths = []
    # for folder_name in folder_names:
    #     for image_name in os.listdir(root_path + folder_name):
    #         image_names.append(image_name.split('.')[0])
    #         image_paths.append("imagenet/train/" + folder_name + "/" + image_name)
    #         local_paths.append(root_path + folder_name + "/" + image_name)

    # from line_profiler import LineProfiler
    from tqdm import tqdm

    # lp = LineProfiler()
    # local_paths = ["/home/wzl/DataSet/imagenet/train/n07695742/n07695742_6957.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n03832673/n03832673_5605.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n03134739/n03134739_12697.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n04536866/n04536866_1100.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n04153751/n04153751_15074.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n03445777/n03445777_6809.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n02504013/n02504013_6379.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n02006656/n02006656_8731.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n02219486/n02219486_30814.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n03888257/n03888257_11187.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n01943899/n01943899_8561.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n03888257/n03888257_11580.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n03662601/n03662601_10228.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n01537544/n01537544_8631.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n02092002/n02092002_15453.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n03792782/n03792782_11989.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n02105251/n02105251_1349.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n02111500/n02111500_6828.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n04228054/n04228054_1074.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n02108089/n02108089_2929.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n04350905/n04350905_24121.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n01806567/n01806567_18296.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n01981276/n01981276_1560.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n01796340/n01796340_16633.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n04435653/n04435653_5390.JPEG",
    #                "/home/wzl/DataSet/imagenet/train/n07930864/n07930864_8612.JPEG"]
    # local_paths = ["/home/wzl/DataSet/imagenet/train/n02894605/n02894605_89121.JPEG"]
    # traverse all images
    for step, image_path in enumerate(tqdm(local_paths[:])):
        # print(step, image_path)
        # load a test image
        image = cv2.imread(image_path)

        ############
        # MaskCut
        ############
        # FeatureMaster = UnionSegMaster(image, tau=0.15, eps=1e-5, UnionSegPatchSize=8, NCutPatchSize=8,
        #                                UnionSegResize=224,
        #                                NCutResize=480, device="cuda:0", pretrained_dino=DinoInstance,
        #                                pretrained_UnionSeg=UnionSegInstance)
        #
        # # obtain features of the image for NCut
        # _, _, ks, _ = FeatureMaster._DINO_features()
        # ks = ks[1:, :]
        # _, cos_similarity_matrix = FeatureMaster._square_distance_matrix(FeatureMaster._norm_features(ks),
        #                                                                  distance_mode="Euclidean",
        #                                                                  feat_nums=ks.shape[0])
        # raw_fore_union = FeatureMaster.returnUnion()
        # MaskUnionSeg = UnionSegMaster(image, tau=0.15, eps=1e-5, UnionSegPatchSize=8, NCutPatchSize=8,
        #                                 UnionSegResize=224,
        #                                 NCutResize=480, device="cpu", UnionSegOutput=raw_fore_union, NCutFeatures=ks,
        #                                 NCutCosSimilarityMatrix=cos_similarity_matrix, use_cupy=False, N=10)
        #
        # masks, fore_union = MaskUnionSeg.maskcut_segmentation(vis=True)

        ############
        # Performance analysis
        ############

        # lp.add_function(MaskUnionSeg._MaskCut)  # 添加第二个待分析函数
        # lp.add_function(ncut)
        # test_func = lp(MaskUnionSeg.maskcut_segmentation)  # 添加分析主函数，注意这里并不是执行函数，所以传递的是是函数名，没有括号。
        # test_func(vis=False)  # 执行主函数，如果主函数需要参数，则参数在这一步传递，例如test_func(参数1, 参数2...)
        # lp.print_stats()

        ############
        # TokenCut
        ############
        try:
            TokenUnionSeg = UnionSegMaster(image, tau=0.2, eps=1e-5, UnionSegPatchSize=8, NCutPatchSize=16,
                                           UnionSegResize=224,
                                           NCutResize=None, device="cpu", use_cupy=False, N=1)
            masks, fore_union = TokenUnionSeg.tokencut_segmentation(vis=True)
        except:
            continue
        # TokenUnionSeg.most_segmentation(vis=True)
        # UnionCut(image, scan_distance_mode="Euclidean", vis=True)

    cv2.destroyAllWindows()
