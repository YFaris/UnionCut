# UnionCut

This is a **temporary, anonymous, and private** GitHub repository for the official implementation of the main functions of UnionCut and UnionSeg, just for your reference.

<p align="center"> <img src='doc/UnionCut_framework.png' align="center" > </p>

_UnionCut receives an image as the input before a series of unit GraphCut is conducted with each patch chosen as the foreground seed. Red rectangles refer to patches selected as foreground seeds, and the blue areas indicate the "foreground" segmented by each unit GraphCut corresponding to the chosen foreground seeds. By aggregating "foreground" areas given by each unit GraphCut, a heat map indicating the foreground union in the image can be obtained. Next, thresholding the intensity-inverted heat map is followed by rectification with a corner prior, resulting in a binary mask of the foreground union in the image._

**Please do not conduct any further development based on the code without the author's permission.
The copyright belongs to the author of the paper titled "Robust Foreground Priors for Enhanced Unsupervised Object Discovery".**

## Install
This repository provides the implementation of UnionCut and UnionSeg, showing examples of combining them with existing unsupervised object discovery (UOD) algorithms (e.g. [TokenCut](https://ieeexplore.ieee.org/document/10224285?denied=) and [MaskCut](https://people.eecs.berkeley.edu/~xdwang/projects/CutLER/)). Clone this repository first and install other dependencies listed in [requirements.txt](/requirements.txt).

```
git clone https://github.com/YFaris/UnionCut.git
```

  
