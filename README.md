# YOVO: You Only Voxelize Once
This is my Bachelor's Thesis at UPT, called **YOVO: You Only Voxelize Once** - 3D Object Reconstruction from a single 2D image. My thesis supervisor was Dr. Eng. Conf. Calin-Adrian Popa.

The nomenclature **_YOVO: You Only Voxelize Once_** derives from the fact that it only uses one input image in order to reconstruct a 3D voxelized representation of the presented object. It is inspired by [Pix2Vox](https://github.com/hzxie/Pix2Vox) and reinvents the Autoencoder module, introducing multi-level feature extraction, leading to multiple volume reconstructions at differente levels of abstractions. Moreover, using a MobileNetV2 backbone, [Mish](https://arxiv.org/abs/1908.08681) activations, [Dropblock](https://arxiv.org/abs/1810.12890) regularization, [Ranger](https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d) optimizer, **YOVO** overcomes the SotA on the ShapeNet subset _Data3D-R2N2_, currently* held by Pix2Vox-A. 
(* _March 2020_)


<div style="text-align:center"><img src="https://i.imgur.com/OC22r87.png"/width = 100%></div>

The YOVO architecture comes in 3 variants:
* **YOVO** : the classic version that introduces mult-level feature processing and a bunch of other techniques
* **YOVO-s** : simplified version that eliminates the *Refiner* and extends the *Decoder*
* **YOVO-e** : extended version that extends both the *Refiner* and the *Decoder*

In-depth details are presented after the _How to Run_ section.

## Results
All three variants of **YOVO** manage to overcome the SotA results of *Pix2Vox-A*.

<div style="text-align:center"><img src="https://i.imgur.com/vcWk5e2.png"/width = 60%></div>

Here are some experimantal results:
<div style="text-align:center"><img src="https://i.imgur.com/GSdIdCU.jpg"/width = 100%></div>

## Dataset

<div style="text-align:center"><img src="https://i.imgur.com/SclIquc.png"/width = 100%></div>

The model can be found at this [link](https://drive.google.com/open?id=17-BY7uKjhebzNxps9hAFU1WNc35JSt3o).

More documentation is coming soon, as the project is still in development and alternative techniques remain to be explored.


This repository is for scientific purposes only and it contains my trial of pushing the results to a new SotA and testing various methods.

#DBU
