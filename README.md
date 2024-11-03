# UnionCut

This is a **temporary, anonymous, and private** GitHub repository for the official implementation of the main functions of UnionCut and UnionSeg, just for your reference.

<p align="center"> <img src='doc/UnionCut_framework.png' align="center" > </p>

_"UnionCut receives an image as the input before a series of unit GraphCut is conducted with each patch chosen as the foreground seed. Red rectangles refer to patches selected as foreground seeds, and the blue areas indicate the "foreground" segmented by each unit GraphCut corresponding to the chosen foreground seeds. By aggregating "foreground" areas given by each unit GraphCut, a heat map indicating the foreground union in the image can be obtained. Next, thresholding the intensity-inverted heat map is followed by rectification with a corner prior, resulting in a binary mask of the foreground union in the image."_

**Please do not conduct any further development based on the code without the author's permission.
The copyright belongs to the author of the paper titled "Robust Foreground Priors for Enhanced Unsupervised Object Discovery".**

## Install
This repository provides the implementation of UnionCut and UnionSeg, showing examples of combining them with existing unsupervised object discovery (UOD) algorithms (e.g. [TokenCut](https://ieeexplore.ieee.org/document/10224285?denied=) and [MaskCut](https://people.eecs.berkeley.edu/~xdwang/projects/CutLER/)). Clone this repository first and install other dependencies (e.g. CuPy, Numpy, torch, and h5py), whose details can be seen in [requirements.txt](/requirements.txt).
```
git clone https://github.com/YFaris/UnionCut.git
```
You may also need to install Detectron2. Please follow its instructions via the [library page](https://github.com/facebookresearch/detectron2) to install.

## Demo
<p align="center"> <img src='doc/demo.png' align="center" > </p>

### UnionCut demo
The core function of UnionCut is implemented in [/UnionCut/DINOinference.py](/UnionCut/DINOinference.py). You can run the demo with the following commands:
```
cd ./UnionCut
python3 DINOinference.py --img-folder [the path of your folder containing test images]
```
This script will create three windows for visualization: the original image, its foreground union given by UnionCut, and the object discovery result by TokenCut+UnionCut. The user can click '_q_' to quit the demo, '_s_' to save the result to _/UnionCut/demo/_, and any other key to the next image.

We also provide a Cython version of UnionCut, which runs 2 times faster than the Python one. It can be run by the command below:
```
python3 DINOinferenceCython.py --img-folder [the path of your folder containing test images]
```

### UnionSeg demo
UnionSeg is trained as a surrogate model of UnionCut, with UnionCut's output on [DUTS-TR](http://saliencydetection.net/duts/) dataset (10,553 images in the training part). Download the dataset first and organize the dataset in your local directory similar to the following style: 
```
Your parent folder for all datasets/
  DUTS/
    DUTS-TR/
      DUTS-TR-Image/*.jpg ...
      DUTS-TR-Mask/*.png ...
    DUTS-TE/
      DUTS-TE-Image/*.jpg ...
      DUTS-TE-Mask/*.png ...
```
We have provided a well-trained UnionSeg, which can be found at _/UnionSeg/module/decoder_weights_niter600.pt_. If you would like to train your own UnionSeg, you can try the following procedure:
1. collect and save UnionCut's output
```
cd ./tools/PseudoMaskGeneration/DUTS-TR
mkdir ./h5
python3 MultiProcess_TokenUnionCut_DUTS_TR_pseudo_mask_generation.py --dataset [the path of your parent folder for all datasets] --process-num [an int value to assign the number of subprocesses]
```
You can choose an appropriate number of subprocesses based on your computational resources by setting _--process-num_ to accelerate UnionCut's execution throughout the whole dataset in parallel. The step may take a few hours. After it is done, an h5 file containing the information of images with corresponding UnionCut's output can be seen in the _h5_ folder created just now.

2. train your UnionSeg
Now you can train your own UnionSeg with the following commands:
```
cd ./UnionSeg
python3 train.py --img-folder [the path of your parent folder for all datasets] --h5-path [the path of the h5 file pregenerated]
```
The training takes only a few minutes. After training, your UnionSeg will be saved at _/UnionSeg/module/decoder_weights_niter600.pt_

Now you can try the demo of UnionSeg:
```
cd ./UnionSeg
python3 inference.py --img-folder [the path of your folder containing test images] --uod-method [MaskCur or TokenCut] --use-cupy [True or False] --N [an int value for MaskCut maximum discovery times per image]
```
Similar to the UnionCut demo, you can see the visualization of the input image, foreground union given by UnionSeg, and discovery result by TokenCut/MaskCut+UnionSeg. You can press '_q_' to quit the demo, '_s_' to save the result to _/UnionSeg/demo/_, and any other key to the next image.

## CutLER+UnionSeg
We use the cookbook of [CutLER](https://github.com/facebookresearch/CutLER/tree/main) with our UnionSeg to train a class-agnostic instance segmentation model. For the installation of CutLER we refer you to the original [CutLER's repository](https://github.com/facebookresearch/CutLER/tree/main). After setting up the official CutLER's implementation, you need to follow the instructions below to combine our UnionSeg with CutLER:
1. copy our UnionSeg package at _/CutLER/third_party/UnionSeg_ under CutLER's corresponding directory: _CutLER/third_party/_
2. replace CutLER's official implementation of MaskCut (_CutLER/maskcut/maskcut.py_) with our MaskCut+UnionSeg (_UnionCut/CutLER/MaskCut/maskcut.py_)
3. replace CutLER's training setting configuration files (_CutLER/cutler/model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_self_train.yaml_ and _CutLER/cutler/model_zoo/configs/CutLER-ImageNet
/cascade_mask_rcnn_R_50_FPN.yaml_) with our version, i.e _/CutLER/configs/cascade_mask_rcnn_R_50_FPN_self_train.yaml_ and _/CutLER/configs/cascade_mask_rcnn_R_50_FPN.yaml_.

Now you can follow the instructions of CutLER to generate pseudo annotations for [ImageNet](https://www.image-net.org/) (you may need to prepare the dataset in advance following the instructions of CutLER). Here are some examples:
> 1. Generate pseudo annotations for ImageNet with our MaskCut+UnionSeg
>> ```
>> cd maskcut
>> python maskcut.py --vit-arch base --patch-size 8 --tau 0.15 --fixed_size 480 --N 3 --num-folder-per-job 1000 --job-index 0 --dataset-path /path/to/dataset/traindir --out-dir /path/to/save/annotations --use-cupy True
>> ```
Note that the argument --N cannot directly change the maximum discovery times of our MaskCut+UnionSeg as it is in CutLER's original implementation. It only affects the filename of generated annotation files. During our 
