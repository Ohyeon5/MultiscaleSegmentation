# Multi-scale feature map induced image segmentation
Deep neural network is mimicking hierachical and feedforward process of human visual cortex. However, it is not a whole story. Human visual system is rather dynamic and recurrsive, therefore, interactive through out different layers.
Such a top-down and bottom-up interactions are seemed to mimicked as a form of residual layers. However, it is unclear how it is explained with regard to human visual processing. 
In current project, features of mutiple scale residual maps are studied, and their integration strategies are studied. Corresponding features and integration strategies are considered with respect to human perceptual features. 
The project is conducted from [Deep Learning Camp Jeju 2018](http://jeju.dlcamp.org/2018/) supported by tensorflow Korea, Google, Kakao-brain, Netmarble, SKT, Element AI, JDC, and Jeju Univ. 

Fully proposed by **Oh-hyeon Choung** *(PhD candidate, EPFL Neuroscence program)*

Main references:
> 1. Lauffs, M. M., Choung, O. H., Öğmen, H., & Herzog, M. H. (2018). Unconscious retinotopic motion processing affects non-retinotopic motion perception. Consciousness and cognition. [(link)](https://www.sciencedirect.com/science/article/pii/S1053810017305421?via%3Dihub)
>2. Shelhamer, E., Long, J., & Darrell, T. (2016). Fully Convolutional Networks for Semantic Segmentation. ArXiv:1605.06211 [Cs]. [(link)](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
>3. Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham. [(link)](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)


## Task: Image semantic segmentation 

examples) 

![alt text](https://github.com/Ohyeon5/MismatchPenaltySegmentation/blob/master/figures/fig_progress/example1.png)
![alt text](https://github.com/Ohyeon5/MismatchPenaltySegmentation/blob/master/figures/fig_progress/example2.png)


**Hypothesis**
1. Feature maps from each convolutional layer include distinct information
2. Depending on it's local/abstract features, could they be integrated using different strategy as human does? 


## Introduction
Human visual system starts from lower visual area and proceed to the higher areas. However, it is not a full story. Our lower visual areas are largely affected by various higher visual area interactively. 

![Retino and Non-retino images][incongOccluded]


## Obejective
1. To 



## Base line model: U-net ()



[incongOccluded]: https://github.com/Ohyeon5/MismatchPenaltySegmentation/blob/master/figures/TPD_blackDisk_cong-incong_occlude.gif
