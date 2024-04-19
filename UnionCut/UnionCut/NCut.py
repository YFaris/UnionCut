"""
Main functions for applying Normalized Cut.
Code adapted from TokenCut: https://github.com/YangtaoWANG95/TokenCut
"""

import numpy as np
from scipy.linalg import eigh
from scipy import ndimage


def ncut(distance_matrix, tau, eps, back_hard=False, fore_hard=False, mode="TokenCut"):
    """
    Implementation of NCut Method.
    Inputs
      distance_matrix: distance matrix of points in the area to be segmented
      indices: abs indices of points in the area to be segmented
      tau: thresold for graph construction
      eps: graph edge weight
    """
    A = distance_matrix.copy()
    if back_hard and fore_hard:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    elif not fore_hard and back_hard:
        A[A < tau] = eps

    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)

    # Print second and third smallest eigenvector
    _, eigenvectors = eigh(D - A, D, subset_by_index=[1, 2])

    # Using average point to compute bipartition
    second_smallest_vec = eigenvectors[:, 0]
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg

    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        bipartition = np.logical_not(bipartition)

    if mode == "TokenCut":
        bipartition = bipartition.reshape([28, 28]).astype(float)
        # predict BBox
        pred, _, objects, cc = detect_box(bipartition, seed, [28, 28], scales=[8, 8],
                                          initial_im_size=[224, 224])  ## We only extract the principal object BBox
        mask = np.zeros([28, 28])
        mask[cc[0], cc[1]] = 1
        return (mask == 1).reshape(-1)
    else:
        return bipartition

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