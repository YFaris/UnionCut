import cv2
import numpy as np
import torch
from CRF.crf import densecrf
import LOST.utils as utils
import LOST.vision_transformer as vits
from torchvision import transforms as pth_transforms
import os, sys
import scipy
from GraphCut.DINOinferenceCython import UnionCut


def preprocess(image, patch_size=16):
    """
    preprocess the cv2 image
    """
    # transfer the image from bgr to rgb
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # pytorch preprocessing
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    torch_image = transform(RGB_image)

    # obtain the size of the feature map
    feat_row, feat_col = int(np.ceil(RGB_image.shape[0] / patch_size)), int(
        np.ceil(RGB_image.shape[1] / patch_size))
    # Padding the image with zeros to fit multiple of patch-size
    paded = torch.zeros((3, feat_row * patch_size, feat_col * patch_size))
    paded[:, :torch_image.shape[1], :torch_image.shape[2]] = torch_image
    torch_image = paded
    return torch_image.unsqueeze(0), feat_row, feat_col  # 1, 3, 224, 224


def load_pretrained_DINO(path, patch_size=16):
    # load the model
    net = vits.vit_small(patch_size=patch_size, num_classes=0)
    # load the model's pre-trained weight
    net.cuda()
    utils.load_pretrained_weights(net, path, checkpoint_key=None, model_name="vit_small",
                                  patch_size=patch_size)
    net.eval()
    return net


def DINO_features(net, image, patch_size=16):
    """
    return features of all patches at the last ViT block
    net: the model
    image: the BGR cv2 image, unnormalized
    """
    # preprocess the image
    torch_image, feat_row, feat_col = preprocess(image, patch_size)
    torch_image = torch_image.cuda()
    # obtain the output embeddings of DINO
    with torch.no_grad():
        features, qs, ks, vs = net.get_last_qkv(torch_image)
    return features, qs, ks, vs, feat_row, feat_col


def LOST_UnionCut(net, image, which_features="k", k=100, crf=True):
    """
    return segmentation mask of the most salient object in the image
    net: the model
    image: the BGR cv2 image, unnormalized
    which_features: the feature to calculate similarity
    K the number of potential seed to be expanded
    """
    # obtain features output by DINO
    _, qs, ks, vs, feat_row, feat_col = DINO_features(net, image)

    # select the feature to calculate the similarity matrix
    if which_features == "k":
        # combine features from multi-head, and remove CLS token
        fs = ks.permute(0, 2, 1, 3).reshape(1, feat_row * feat_col + 1, -1).squeeze(0)[1:, :]  # 784, 384
    elif which_features == "q":
        # combine features from multi-head, and remove CLS token
        fs = qs.permute(0, 2, 1, 3).reshape(1, feat_row * feat_col + 1, -1).squeeze(0)[1:, :]  # 784, 384
    elif which_features == "v":
        # combine features from multi-head, and remove CLS token
        fs = vs.permute(0, 2, 1, 3).reshape(1, feat_row * feat_col + 1, -1).squeeze(0)[1:, :]  # 784, 384
    fore_union = UnionCut(image)
    pad = np.zeros((feat_row * 16, feat_col * 16), dtype=np.uint8)
    pad[: image.shape[0], : image.shape[1]] = cv2.resize(fore_union, (image.shape[1], image.shape[0]), cv2.INTER_NEAREST)
    fore_union = cv2.resize(pad, (feat_col, feat_row), cv2.INTER_NEAREST)
    fore_union = torch.from_numpy(fore_union).cuda().reshape(-1)
    pred, _, _, _, raw_mask = lost_unioncut(fs, fore_union, (feat_row, feat_col), (16, 16), (3, image.shape[0], image.shape[1]), k_patches=k)
    mask = cv2.resize(raw_mask, (feat_col * 16, feat_row * 16), cv2.INTER_NEAREST)[: image.shape[0], : image.shape[1]]
    if crf:
        # apply crf
        mask = densecrf(image, mask.astype(np.uint8)).astype(np.uint8)
    return pred, mask


def lost_unioncut(feats, fore_union, dims, scales, init_image_size, k_patches=100):
    """
    Implementation of LOST method.
    Inputs
        feats: the pixel/patche features of an image
        dims: dimension of the map from which the features are used
        scales: from image to map scale
        init_image_size: size of the image
        k_patches: number of k patches retrieved that are compared to the seed at seed expansion
    Outputs
        pred: box predictions
        A: binary affinity matrix
        scores: lowest degree scores for all patches
        seed: selected patch corresponding to an object
    """
    # Compute the similarity
    A = (feats @ feats.T)

    # # Compute the inverse degree centrality measure per patch
    # sorted_patches, scores = patch_scoring(A)
    # sorted_patches = sorted_patches[fore_union[sorted_patches] == 1]
    # # Select the initial seed
    # seed = sorted_patches[0]

    # Compute the inverse degree centrality measure per patch
    sorted_patches, scores = patch_scoring(A)
    seed_sorted_patches = sorted_patches[fore_union[sorted_patches] == 1]
    # Select the initial seed
    seed = seed_sorted_patches[0]

    # Seed expansion
    potentials = sorted_patches[:k_patches]
    # potentials = sorted_patches[:]
    similars = potentials[A[seed, potentials] > 0.0]
    M = torch.sum(A[similars, :], dim=0)

    # Box extraction
    pred, _, raw_mask = detect_box(
        M, seed, dims, scales=scales, initial_im_size=init_image_size[1:]
    )

    return np.asarray(pred), A, scores, seed, raw_mask


def patch_scoring(M, threshold=0.):
    """
    Patch scoring based on the inverse degree.
    """
    # Cloning important
    A = M.clone()

    # Zero diagonal
    A.fill_diagonal_(0)

    # Make sure symmetric and non nul
    A[A < 0] = 0
    C = A + A.t()

    # Sort pixels by inverse degree
    cent = -torch.sum(A > threshold, dim=1).type(torch.float32)
    sel = torch.argsort(cent, descending=True)

    return sel, cent


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
        raise ValueError("The seed is in the background component.")

    # Find box
    mask = np.where(labeled_array == cc)
    # initialize mask
    raw_mask = np.zeros(dims, dtype=np.uint8)
    raw_mask[mask[0], mask[1]] = 1
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

    return pred, pred_feats, raw_mask


def visualization(masks):
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


if __name__ == "__main__":
    # load the model
    net = load_pretrained_DINO(path="./dino_deitsmall16_pretrain.pth")
    image_paths = [
        "/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/" + filename for filename in
        os.listdir("/media/wzl/T7/DataSet/VOC/VOC2007/VOCdevkit/VOC2007/JPEGImages/")]

    # image_paths = ["/home/wzl/DataSet/imagenet/train/n07695742/n07695742_6957.JPEG",
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
    for image_path in image_paths[:]:
        # load a test image
        image = cv2.imread(image_path)
        # image = cv2.imread("./CMS_livingroom.PNG")
        # obtain mask
        pred, mask = LOST_UnionCut(net, image)
        cv2.rectangle(
            image,
            (int(pred[0]), int(pred[1])),
            (int(pred[2]), int(pred[3])),
            (255, 0, 0), 3,
        )
        cv2.imshow("image", image)
        cv2.imshow("mask", 255 * mask)
        # waiting for user interaction
        pressedKey = cv2.waitKey(0) & 0xFF
        if pressedKey == ord('q'):
            # press q to quit
            break
        elif pressedKey == ord('s'):
            # save the figue and mask
            cv2.imwrite("./demo/" + image_path.split("/")[-1], image)
            cv2.imwrite("./demo/" + image_path.split("/")[-1].split(".")[0] + "_mask.jpg", vis)
    cv2.destroyAllWindows()
