"""
Main functions for applying Normalized Cut.
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from crf import densecrf
from scipy import ndimage
import cupy as cp


def ncut(distance_matrix, tau, eps, feat_row=28, feat_col=28, left_indices=None, use_cupy=False):
    """
    Implementation of NCut Method.
    Inputs
      distance_matrix: distance matrix of points in the area to be segmented
      indices: abs indices of points in the area to be segmented
      tau: thresold for graph construction
      eps: graph edge weight
      row, col: size of the graph
      img_row, img_col: size of the image
      patch_size: patch size of the ViT
      left_indices: indices of unremoved patches
    """
    if not use_cupy:
        A = distance_matrix > tau
        A = np.where(A.astype(float) == 0, eps, A)

        d_i = np.sum(A, axis=1)
        D = np.diag(d_i)

        # Print second and third smallest eigenvector
        _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])

        # Using average point to compute bipartition
        second_smallest_vec = eigenvectors[:, 0]
        avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
        bipartition = second_smallest_vec > avg

        seed = np.argmax(np.abs(second_smallest_vec))

        if left_indices is not None and not np.min(left_indices):
            # map the bipartition on the feature map
            indices = np.arange(feat_row*feat_col)
            mapped_bipartition = np.zeros(feat_row*feat_col, dtype=np.bool_)
            mapped_bipartition[indices[left_indices]] = bipartition
            bipartition = mapped_bipartition

        # bipartition = bipartition.reshape([feat_row, feat_col]).astype(float)
        # # predict BBox
        # pred, _, objects, cc = detect_box(bipartition, seed, [feat_row, feat_col], scales=[patch_size, patch_size],
        #                                   initial_im_size=[img_row, img_col])  ## We only extract the principal object BBox
        # mask = np.zeros([feat_row, feat_col])
        # mask[cc[0], cc[1]] = 1
        return bipartition, seed, second_smallest_vec
    else:
        A = distance_matrix > tau
        A = cp.where(A.astype(float) == 0, eps, A)

        d_i = cp.sum(A, axis=1)
        D = cp.diag(d_i)

        A = D - A  # 重定义 A 为 D - A
        D_sqrt_inv = cp.linalg.inv(cp.linalg.cholesky(D))  # 计算 D 的 Cholesky 分解并求逆

        # 将 A 转换为标准形式
        A_transformed = D_sqrt_inv @ A @ D_sqrt_inv.T

        # 使用标准特征值解法
        _, eigenvectors_transformed = cp.linalg.eigh(A_transformed)

        # 变换回原始特征向量
        eigenvectors = D_sqrt_inv.T @ eigenvectors_transformed

        # _, eigenvectors = cp.linalg.eigh(D - A, D)
        eigenvectors = eigenvectors[:, [1, 2]]

        # Using average point to compute bipartition
        second_smallest_vec = eigenvectors[:, 0]
        avg = cp.mean(second_smallest_vec)
        bipartition = second_smallest_vec > avg

        seed = cp.argmax(cp.abs(second_smallest_vec))

        if left_indices is not None and not np.min(left_indices):
            # map the bipartition on the feature map
            indices = cp.arange(feat_row * feat_col)
            mapped_bipartition = cp.zeros(feat_row * feat_col, dtype=cp.bool_)
            mapped_bipartition[indices[left_indices]] = bipartition
            bipartition = mapped_bipartition
        return bipartition.get(), seed.get(), second_smallest_vec.get()

def detect_box(bipartition, seed, dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]

    if principle_object:
        mask = np.where(objects == cc)
        # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError

def judge_bipartition(sub_fore_indices1, sub_fore_indices2, sub_fore_mask1, sub_fore_mask2, attention_map, area_ratio_thresh=0.1, attention_thresh=1):

    def _area_ratio(mask1, mask2):
        # between the area of two sub-masks
        # obtain areas of two masks
        area_1 = np.sum(mask1)
        area_2 = np.sum(mask2)
        # obtain ratio of area
        ratio = min(area_1, area_2) / (area_1 + area_2)
        return ratio

    def _attention_connection_ratio(fore_indices1, fore_indices2, attention_mtx):
        attention_slice_one2two = attention_mtx[fore_indices1, :][:, fore_indices2]
        attention_slice_two2one = attention_mtx[fore_indices2, :][:, fore_indices1]
        # obtain the number of points in area2 focused by points in area1
        num_one2two = np.sum(attention_slice_one2two, axis=0)
        num_one2two[num_one2two > 1] = 1
        num_one2two = np.sum(num_one2two)
        # obtain the number of points in area1 focused by points in area2
        num_two2one = np.sum(attention_slice_two2one, axis=0)
        num_two2one[num_two2one > 1] = 1
        num_two2one = np.sum(num_two2one)
        return (num_one2two / len(fore_indices2)) * (num_two2one / len(fore_indices1))

    # calculate area ratio
    area_ratio = _area_ratio(sub_fore_mask1, sub_fore_mask2)
    # calculate attention connection ratio
    connection_ratio = _attention_connection_ratio(sub_fore_indices1, sub_fore_indices2, attention_map)
    # summarize the two ratio to judge thi quality of the bipartition
    if area_ratio > area_ratio_thresh and connection_ratio < attention_thresh:
        return True
    else:
        return False