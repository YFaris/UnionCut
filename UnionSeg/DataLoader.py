"""
Dataset functions for applying Normalized Cut.
Code adapted from SelfMask: https://github.com/NoelShin/selfmask
Modified by the author of UnionCut
"""

import os
from typing import Optional, Tuple, Union
import numpy as np
import torch
from augmentations import geometric_augmentations, photometric_augmentations
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from geometric_transforms import resize
import h5py

NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def unnormalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Code borrowed from STEGO: https://github.com/mhamilton723/STEGO
    """
    image2 = torch.clone(image)
    for t, m, s in zip(image2, mean, std):
        t.mul_(s).add_(m)

    return image2


class UnionSegDataset(Dataset):
    def __init__(
            self,
            img_dir: str,
            h5_path: str
    ) -> None:
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(UnionSegDataset, self).__init__()
        self.use_aug = True  # use augmentation
        self.img_dir = img_dir  # root path of images
        self.h5_path = h5_path  # path of the h5 file
        # load the h5 file
        file = h5py.File(h5_path, "r")
        # initialize image names of the dataset
        method_group = file["TokenGraphCut"]
        self.image_names = list(method_group.keys())
        # close the h5 file
        file.close()
        self.ignore_index = -1
        self.mean = NORMALIZE.mean
        self.std = NORMALIZE.std
        self.to_tensor_and_normalize = T.Compose([T.ToTensor(), NORMALIZE])
        self.normalize = NORMALIZE

        self._set_aug()

    def get_init_transformation(self):
        t = T.Compose([T.ToTensor(), NORMALIZE])
        t_nonorm = T.Compose([T.ToTensor()])
        return t, t_nonorm

    def _set_aug(self):
        """
        Set augmentation based on config.
        """
        self.cropping_strategy = "random_scale"
        self.scale_range = [0.1, 3.0]
        self.crop_size = 224
        self.center_crop_transforms = T.Compose(
            [
                T.CenterCrop((self.crop_size, self.crop_size)),
                T.ToTensor(),
            ]
        )
        self.center_crop_only_transforms = T.Compose(
            [T.CenterCrop((self.crop_size, self.crop_size)), T.PILToTensor()]
        )

        self.proba_photometric_aug = 0.5

        self.random_color_jitter = False
        self.random_grayscale = False
        self.random_gaussian_blur = True

    def _preprocess_data_aug(
            self,
            image: Image.Image,
            mask: Image.Image,
            ignore_index: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare data in a proper form for either training (data augmentation) or validation."""

        # resize to base size
        image = resize(
            image,
            size=self.crop_size,
            edge="shorter",
            interpolation="bilinear",
        )
        mask = resize(
            mask,
            size=self.crop_size,
            edge="shorter",
            interpolation="bilinear",
        )

        if not isinstance(mask, torch.Tensor):
            mask: torch.Tensor = torch.tensor(np.array(mask))

        random_scale_range = None
        random_crop_size = None
        random_hflip_p = None
        if self.cropping_strategy == "random_scale":
            random_scale_range = self.scale_range
        elif self.cropping_strategy == "random_crop":
            random_crop_size = self.crop_size
        elif self.cropping_strategy == "random_hflip":
            random_hflip_p = 0.5
        elif self.cropping_strategy == "random_crop_and_hflip":
            random_hflip_p = 0.5
            random_crop_size = self.crop_size

        if random_crop_size or random_hflip_p or random_scale_range:
            image, mask = geometric_augmentations(
                image=image,
                mask=mask,
                random_scale_range=random_scale_range,
                random_crop_size=random_crop_size,
                ignore_index=ignore_index,
                random_hflip_p=random_hflip_p,
            )

        if random_scale_range:
            # resize to (self.crop_size, self.crop_size)
            image = resize(
                image,
                size=self.crop_size,
                interpolation="bilinear",
            )
            mask = resize(
                mask,
                size=(self.crop_size, self.crop_size),
                interpolation="bilinear",
            )

        image = photometric_augmentations(
            image,
            random_color_jitter=self.random_color_jitter,
            random_grayscale=self.random_grayscale,
            random_gaussian_blur=self.random_gaussian_blur,
            proba_photometric_aug=self.proba_photometric_aug,
        )

        # to tensor + normalize image
        image = self.to_tensor_and_normalize(image)

        return image, mask

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx):
        # load the h5 file
        file = h5py.File(self.h5_path, "r")
        method_group = file["TokenGraphCut"]
        # obtain the image path
        image_path_key = self.image_names[idx] + "/path"
        img_path = self.img_dir + method_group[image_path_key][()].decode('utf-8')
        with open(img_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        # load the union mask of the image
        mask_gt = method_group[self.image_names[idx] + '/union_crf'][:].astype(np.uint8)[None, :, :]
        img_t, mask_gt = self._preprocess_data_aug(
            image=img, mask=mask_gt, ignore_index=self.ignore_index
        )
        mask_gt = mask_gt.squeeze(0)

        img_init = unnormalize(img_t)
        # close the h5 file
        file.close()
        return img_t, img_init, mask_gt, img_path


if __name__ == "__main__":
    import torch.utils.data as Data
    import cv2

    torch_dataset = UnionSegDataset("/home/wzl/DataSet/", "/media/wzl/T7/PhysicalImageAugmentation/PseudoMaskGenration/DUTS/h5/TokenGraphCutN1_tao0.2_DUTS_Train_ViT_S_16_flexible_8_fixed.h5")
    BATCH_SIZE = 1
    # 把dataset放入DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=1,  # 多线程来读数据
    )
    for step, (img_t, img_init, mask_gt, img_path) in enumerate(loader):
        # denormalize the input image for visualization
        image = np.transpose(img_init[0].cpu().numpy(), (1, 2, 0))
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        instance_label_masks = 255 * mask_gt[0].numpy().astype(np.uint8)
        cv2.imshow("image", image)
        cv2.imshow("mask", instance_label_masks)
        # waiting for user interaction
        pressedKey = cv2.waitKey(0) & 0xFF
        if pressedKey == ord('q'):
            # press q to quit
            cv2.destroyAllWindows()
            break
        else:
            continue
