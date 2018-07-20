# MismatchPenaltySegmentation
Mimicking human parallel visual information processing system (mismatch penalty) for Image semantic segmentation, using multiple feature maps. The project will be presented in [Deep Learning Camp Jeju 2018](http://jeju.dlcamp.org/2018/). 

Fully proposed by **Oh-hyeon Choung** *(PhD candidate, EPFL Neuroscence program)*

Main references:
> 1. Lauffs, M. M., Choung, O. H., Öğmen, H., & Herzog, M. H. (2018). Unconscious retinotopic motion processing affects non-retinotopic motion perception. Consciousness and cognition. [(link)](https://www.sciencedirect.com/science/article/pii/S1053810017305421?via%3Dihub)
>2. Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., & Berg, A. C. (2016). SSD: Single Shot MultiBox Detector. ArXiv:1512.02325 [Cs], 9905, 21–37. [(link)](https://arxiv.org/abs/1512.02325),[(GitHub)](https://github.com/weiliu89/caffe/tree/ssd)
>3. Shelhamer, E., Long, J., & Darrell, T. (2016). Fully Convolutional Networks for Semantic Segmentation. ArXiv:1605.06211 [Cs]. [(link)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)


## Introduction
Human visual system starts from lower visual area and proceed to the higher areas. However, it is not a full story. Our lower visual areas are largely affected by various higher visual area interactively. 

![Retino and Non-retino images][incongOccluded]


## Obejective
1. To 



[incongOccluded]: https://github.com/Ohyeon5/MismatchPenaltySegmentation/blob/master/figures/TPD_blackDisk_cong-incong_occlude.gif


# Multi-scale feature map induced image segmentation

**Hypothesis**
1. Feature maps from each convolutional layer include distinct information
2. Depending on it's local/abstract features, could they be integrated using different strategy as human does? 


## Base line model: U-net ()




## Task: Image semantic segmentation 

examples) 

![alt text](https://github.com/Ohyeon5/MismatchPenaltySegmentation/blob/master/figures/fig_progress/example1.png)
![alt text](https://github.com/Ohyeon5/MismatchPenaltySegmentation/blob/master/figures/fig_progress/example2.png)
