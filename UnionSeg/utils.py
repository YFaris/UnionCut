# Code borrowed from FOUND: https://github.com/valeoai/FOUND/tree/main
import torch
import numpy as np
import scipy


def entropy_tensor(vals):
    _, counts = torch.unique(vals, return_counts=True)
    freq = counts / torch.sum(counts)
    ent = -1 * torch.sum(freq *
                         torch.log(freq) / torch.log(torch.tensor([2.])).to(freq.device))
    return ent.item()


def compute_block_entropy(map_, poolers):
    with torch.no_grad():
        f = [l(map_.unsqueeze(0).unsqueeze(0).cuda()).reshape(-1) for l in poolers]
    ents = [entropy_tensor(l) for l in f]
    return ents


def refine_mask(A, seed, dims):
    w_featmap, h_featmap = dims

    correl = A.reshape(w_featmap, h_featmap).float()

    # Compute connected components
    labeled_array, num_features = scipy.ndimage.label(correl.cpu().numpy() > 0.0)

    # Find connected component corresponding to the initial seed
    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))]

    if cc == 0:
        return [], []
    mask = A.reshape(w_featmap, h_featmap).cpu().numpy()
    mask[labeled_array != cc] = -1

    return torch.tensor(mask)


def detect_box(A, seed, dims, initial_im_size=None, scales=None):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims

    correl = A.reshape(w_featmap, h_featmap).float()

    # Compute connected components
    labeled_array, num_features = scipy.ndimage.label(correl.cpu().numpy() > 0.0)

    # Find connected component corresponding to the initial seed
    cc = labeled_array[np.unravel_index(seed.cpu().numpy(), (w_featmap, h_featmap))]

    # Should not happen with LOST
    if cc == 0:
        # raise ValueError("The seed is in the background component.")
        return [], []

    # Find box
    mask = np.where(labeled_array == cc)
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

    return [pred], [pred_feats]
