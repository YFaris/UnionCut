import torch.nn as nn
import torch
from GraphCut.UnionSeg.model import UnionSeg
import numpy as np
import GraphCut.utils as utils
import GraphCut.vision_transformer as vits
import cv2
from torchvision import transforms as pth_transforms
import torch.nn.functional as F
from GraphCut.NCut import ncut
from CRF.crf import densecrf


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
    def __init__(self, image, tau=0.2, eps=1e-5, UnionSegPatchSize=8, NCutPatchSize=16, UnionSegResizeMode="fixed",
                 NCutResizeMode="flexible", device="cpu", pretrained_dino=None, pretrained_UnionSeg=None, theta=0.5):
        """
        image: image to be segmented
        tau: threshold for attention map construction
        eps: graph edge weight of non-relative links
        patch_size: size of the patch
        resize_mode: the input size of the image, fixed -> 224x224; flexible - > ratio does not change
        device: cpu or cuda
        pretrained_dino: pretrained dino
        pretrained_UnionSeg: pretrained UnionSeg
        theta: threshold for valid masks
        """
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
        self.N = 1
        # initialize the patch size
        self.UnionSegPatchSize = UnionSegPatchSize
        self.NCutPatchSize = NCutPatchSize
        # initialize preprocess mode of resize
        self.UnionSegResizeMode = UnionSegResizeMode
        self.NCutResizeMode = NCutResizeMode
        # load model
        if pretrained_UnionSeg is not None:
            self.UnionSegModel = pretrained_UnionSeg
        else:
            self.UnionSegModel = self.load_pretrained_UnionSeg()  # UnionSeg model
        if pretrained_dino is not None:
            self.DinoModel = pretrained_dino
        else:
            self.DinoModel = self.load_pretrained_DINO(NCutPatchSize)
        self.sigmoid = nn.Sigmoid()
        # move models to test device
        self.UnionSegModel.to(self.device)
        self.DinoModel.to(self.device)
        self.sigmoid.to(self.device)
        # obtain the size of the feature map
        if self.UnionSegResizeMode == "fixed":
            self.UnionSeg_feat_row, self.UnionSeg_feat_col = 224 // UnionSegPatchSize, 224 // UnionSegPatchSize
        else:
            self.UnionSeg_feat_row, self.UnionSeg_feat_col = int(np.ceil(self.row / self.UnionSegPatchSize)), int(
                np.ceil(self.col / self.UnionSegPatchSize))
        if self.NCutResizeMode == "fixed":
            self.NCut_feat_row, self.NCut_feat_col = 224 // NCutPatchSize, 224 // NCutPatchSize
        else:
            self.NCut_feat_row, self.NCut_feat_col = int(np.ceil(self.row / NCutPatchSize)), int(
                np.ceil(self.col / NCutPatchSize))

    @staticmethod
    def load_pretrained_DINO(patch_size):
        # load the model
        if patch_size == 8:
            net = vits.vit_small(patch_size=8, num_classes=0)
            # load the model's pre-trained weight
            # net.cuda()
            utils.load_pretrained_weights(net,
                                          "/media/wzl/T7/PhysicalImageAugmentation/LOST/dino_deitsmall8_pretrain.pth",
                                          checkpoint_key=None,
                                          model_name="vit_small",
                                          patch_size=8)
        elif patch_size == 16:
            net = vits.vit_small(patch_size=16, num_classes=0)
            # load the model's pre-trained weight
            # net.cuda()
            utils.load_pretrained_weights(net,
                                          "/media/wzl/T7/PhysicalImageAugmentation/TokenCut/dino_deitsmall16_pretrain.pth",
                                          checkpoint_key=None,
                                          model_name="vit_small",
                                          patch_size=16)
        net.eval()
        return net

    @staticmethod
    def _preprocess(image, patch_size, resize_mode, feat_row, feat_col):
        """
        preprocess the cv2 image
        mode: fixed -> 224 * 224; flexible -> ratio does not change
        """
        # transfer the image from bgr to rgb
        RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if resize_mode == "fixed":
            # resize the image to 224x224
            RGB_image = cv2.resize(RGB_image, (224, 224))
        elif resize_mode == "flexible":
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
                                       resize_mode=self.NCutResizeMode,
                                       feat_row=self.NCut_feat_row,
                                       feat_col=self.NCut_feat_col)

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
            "/media/wzl/T7/PhysicalImageAugmentation/GraphCut/UnionSeg/module/decoder_weights_niter600.pt")
        UnionSegModel.eval()
        return UnionSegModel

    @staticmethod
    def _square_distance_matrix(features, distance_mode, feat_nums):
        """
        calculate squared distance matrix of each point
        params:
            features: (self.feat_row * self.feat_col, 384)
        """
        # resize features to -1, 384
        flatten_features = features.reshape(-1, 384)
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
            differences = (flatten_features.reshape(feat_nums, 1, 384) - flatten_features.reshape(1, feat_nums,
                                                                                                  384)).reshape(-1, 384)
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

    def TokenCut(self, features, fore_union, left_indices):
        features = features[left_indices, :]
        # initialize wait list
        waitlist = []
        # calculate distance matrix
        _, cos_similarity_matrix = self._square_distance_matrix(self._norm_features(features),
                                                                distance_mode="Euclidean",
                                                                feat_nums=features.shape[0])
        # TokenCut
        for _ in range(self.N):
            # NCut
            try:
                bipartition = ncut(cos_similarity_matrix, self.tau, self.eps, back_hard=True, fore_hard=True,
                                   mode="TokenCut", feat_row=self.NCut_feat_row, feat_col=self.NCut_feat_col,
                                   img_row=self.NCut_feat_row * 16, img_col=self.NCut_feat_col * 16,
                                   patch_size=self.NCutPatchSize, left_indices=left_indices)
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
                if IoU(sub_fore_mask, waitlist[-1]) > 0.5 or \
                        self._check_num_fg_corners(cv2.erode(sub_fore_mask,
                                                             kernel=cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                                              (5, 5)))) >= 3:
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
        return waitlist

    def returnUnion(self, vis=False):
        """
        return the foreground union segmented by UnionSeg
        """
        # preprocess the image
        torch_image = self._preprocess(image=self.image,
                                       patch_size=self.UnionSegPatchSize,
                                       resize_mode=self.UnionSegResizeMode,
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
        return fore_union

    def TokenCut_segmentation(self, small_thresh=0.01, vis=False):
        """
        TokenCut + UnionSeg segmentation
        """
        # obtain the foreground Union
        raw_fore_union = self.returnUnion()
        fore_union = densecrf(self.image, raw_fore_union).astype(np.uint8)
        fore_union = cv2.resize(fore_union, (self.NCut_feat_col, self.NCut_feat_row), cv2.INTER_NEAREST)

        # obtain features of the image for NCut
        _, _, ks, _ = self._DINO_features()
        ks = ks[1:, :]
        # initialize indices of unremoved patches
        left_indices = np.ones(ks.shape[0], dtype=np.bool_)
        for _ in range(5):
            # hierarchical NCut
            waitlist = self.TokenCut(ks, fore_union, left_indices)
            # initialize result list
            masks = []
            # remove false positive
            for mask in waitlist:
                # only keep the mask covered by foreground union
                print(Precision(mask, fore_union))
                if Precision(mask, fore_union) > self.theta:
                    masks.append(mask)
            if len(masks) == 0:
                # no satisfied masks discovered
                waitlist.sort(key=lambda x: np.sum(x))
                # remove the content of the first mask in the wait list from the image
                left_indices[((waitlist[0] - waitlist[0] * fore_union) == 1).reshape(-1)] = False
            else:
                break
        # if no satisfied masks, make the foreground union given by UnionSeg as the final output
        if len(masks) == 0:
            masks.append(fore_union)

        # apply crf
        crf_masks = []
        for mask in masks:
            crf_masks.extend(self._disconnection_detection(mask))
        # if no satisfied masks, make the foreground union given by UnionSeg as the final output
        crf_fore = densecrf(self.image, raw_fore_union).astype(np.uint8)
        if len(crf_masks) == 0:
            if np.sum(crf_fore) == 0:
                crf_masks.append(
                    cv2.resize(raw_fore_union, (self.image.shape[1], self.image.shape[0]), cv2.INTER_NEAREST))
            else:
                crf_masks.append(crf_fore)
        masks = np.stack(crf_masks)
        # remove too small masks
        masks = masks[np.sum(masks, axis=(1, 2)) / np.sum(
            cv2.resize(raw_fore_union, (self.image.shape[1], self.image.shape[0]), cv2.INTER_NEAREST)) > small_thresh]
        # make sure one mask is returned
        if len(masks) == 0:
            if np.sum(fore_union) != 0:
                masks = np.stack([densecrf(self.image, raw_fore_union).astype(np.uint8)])
            else:
                masks = np.stack(
                    [cv2.resize(raw_fore_union, (self.image.shape[1], self.image.shape[0]), cv2.INTER_NEAREST)])
        if vis:
            cv2.imshow('image', self.image)
            cv2.imshow("fore total", 255 * fore_union)
            # print(len(background_ids), max_thresh)
            # plt.show()
            colorful_masks = self._visualization(masks)
            cv2.imshow("masks", colorful_masks)
            cv2.waitKey(0)
        return masks, fore_union

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

    def returnUnionSegBbox(self, crf=True, vis=False):
        """
        return the bbox of the largest connected area
        """
        # obtain the foreground Union
        fore_union = self.returnUnion()
        if crf:
            # conduct crf
            fore_union = densecrf(self.image, fore_union).astype(np.uint8)
            # open
            kernel = np.ones((5, 5), np.uint8)
            fore_union = cv2.morphologyEx(fore_union, cv2.MORPH_OPEN, kernel, iterations=1)
        # obtain all isolated masks
        masks = self._disconnection_detection(fore_union, crf)
        # sort the mask by area in descending order
        masks.sort(key=lambda x: np.sum(x), reverse=True)
        # obtain the bbox of the first mask
        xmin, ymin, xmax, ymax = self.find_bbox(masks[0])
        # visualization
        if vis:
            image = cv2.rectangle(self.image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
            cv2.imshow("image", image)
            cv2.waitKey(0)
        return xmin, ymin, xmax, ymax


if __name__ == "__main__":
    import os

    image_paths = ["/home/wzl/DataSet/ECSSD/images/0371.jpg"] + [
        "/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/" + filename for filename in
        os.listdir("/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/")]

    # initialize model
    UnionSeg = UnionSegMaster.load_pretrained_UnionSeg()
    Dino = UnionSegMaster.load_pretrained_DINO(patch_size=16)
    # traverse all images
    for step, image_path in enumerate(image_paths[:]):
        print(step, image_path)
        # load a test image
        image = cv2.imread(image_path)
        # image = cv2.imread("/home/wzl/TokenCut/DINO/CMS_livingroom.PNG")
        # image = cv2.imread("./berlin_000011_000019_leftImg8bit.png")
        # image = cv2.resize(image, (int(0.5 * image.shape[1]), int(0.5 * image.shape[0])), cv2.INTER_AREA)
        # graph_cut = GraphCutMaster(image)
        # masks = graph_cut.segmentation()
        TokenUnionSeg = UnionSegMaster(image, tau=0.2, eps=1e-5, UnionSegPatchSize=8, NCutPatchSize=16,
                                       UnionSegResizeMode="fixed",
                                       NCutResizeMode="flexible", device="cpu", pretrained_dino=Dino,
                                       pretrained_UnionSeg=UnionSeg)
        # foreground_union = TokenUnionSeg.returnUnion(vis=True)
        # xmin, ymin, xmax, ymax = TokenUnionSeg.returnUnionSegBbox(vis=True)
        masks, fore_union = TokenUnionSeg.TokenCut_segmentation(vis=True)
        # UnionCut(image, scan_distance_mode="Euclidean", vis=True)
    cv2.destroyAllWindows()
