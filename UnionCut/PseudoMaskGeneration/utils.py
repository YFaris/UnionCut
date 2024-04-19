import cv2
import torch
import UnionCut.utils as utils
import UnionCut.vision_transformer as vits
from torchvision import transforms as pth_transforms
import numpy as np

def preprocess(image):
    """
    preprocess the cv2 image
    """
    # transfer the image from bgr to rgb
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize the image to 224x224
    RGB_image = cv2.resize(RGB_image, (224, 224))
    # pytorch preprocessing
    transform = pth_transforms.Compose([
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    torch_image = transform(RGB_image)
    return torch_image.unsqueeze(0)  # 1, 3, 224, 224


def load_pretrained_DINO(path):
    # load the model
    net = vits.vit_small(patch_size=8, num_classes=0)
    # load the model's pre-trained weight
    net.cuda()
    utils.load_pretrained_weights(net, path, checkpoint_key=None, model_name="vit_small",
                                  patch_size=8)
    net.eval()
    return net


def DINO_features(net, image):
    """
    return features of all patches at the last ViT block
    net: the model
    image: the BGR cv2 image, unnormalized
    """
    # preprocess the image
    torch_image = preprocess(image)
    torch_image = torch_image.cuda()
    # obtain the output embeddings of DINO
    with torch.no_grad():
        features, qs, ks, vs = net.get_last_qkv(torch_image)
    return features, qs, ks, vs

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
        x_mass, y_mass = np.sum((binary_image*coord).reshape(2, -1), axis=1) / np.sum(binary_image)
        return [x_mass, y_mass, w, h]
    else:
        # return normal bbox
        return [xmin, ymin, xmax, ymax]