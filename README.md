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
You can choose an appropriate number of subprocesses based on your computational resources by setting _--process-num_ to accelerate UnionCut's execution throughout the whole dataset in parallel. After it is done, an h5 file containing the information of images with corresponding UnionCut's output can be seen in the _h5_ folder created just now.

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
