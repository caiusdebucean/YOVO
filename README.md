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

The [ShapeNet](https://www.shapenet.org/) dataset is used to sample the *Data 3D-R2N2* subset, which is used in the experiments. The download links are avaliable below:

- ShapeNet rendering images: [Renderings](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz)
- ShapeNet voxelized models: [3D voxelized models](http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz)


## Prerequisites

####  Pretrained weights

***YOVO*** original model can be found at: [YOVO](https://drive.google.com/file/d/12QLuJ1oKtraTH-0ZFpmNo5MUSfrVbOyG/view?usp=sharing).
***YOVO-s*** and ***YOVO-e*** model can be found at: [YOVO-s / YOVO-e](https://drive.google.com/file/d/1rQEtEAKBgtGVoqzFck6PWvy_Bsoo7cuv/view?usp=sharing).

#### Clone the Code Repository

```
git clone https://github.com/caiusdebucean/YOVO.git

```

#### Install Python Denpendencies

```
cd YOVO
pip install -r requirements.txt
```

#### Update Settings in `config.py`
The code is heavily inspired by Pix2Vox. Credits to the original creators.

**Dataset location:**

```
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/path/to/Datasets/ShapeNet/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/path/to/Datasets/ShapeNet/ShapeNetVox32/%s/%s/model.binvox'
```

**YOVO Architecture:**

```
__C.NETWORK.YOVO_VERSION = 'classic' #[classic, simple, extended, custom]
```

**Visualizing results:**

```
__C.TEST.VIEW_KAOLIN                        = True  # Rendering with kaolin. This should be done locally, not through ssh
__C.TEST.SAVE_RENDERED_IMAGE                = True # Save the input preprocessed image containing the object
__C.TEST.NO_OF_RENDERS                      = 1 # How many examples/class to be saved for visualization during test/validation
__C.TEST.SAVE_GIF                           = True # Save a GIF of 360 rotating volume for the saved objects
__C.TEST.RENDER_THRESHOLD                   = 0.85 # How confident should the saved predictions be
__C.TEST.GENERATE_MULTILEVEL_VOLUMES        = True # Generate the reconstructed volumes at the autoencoder level
```

## How to run

To train **YOVO**, run following command in the _root_ folder:
```
python3 runner.py --name custom_name
```

To test **YOVO**, run following command in the _root_ folder:
```
python3 runner.py --name custom_name --test --weights=/path/model.pth
```


## Detailed information
The full architecture can be seen here:

<div style="text-align:center"><img src="https://i.imgur.com/SclIquc.png"/width = 100%></div>



## License

This project is open sourced under MIT license, and is for scientific purposes only, containing my trials of pushing the results to a new SotA and testing various methods.

## End Note

More documentation is coming soon, as the project is still in development and alternative techniques/ablation studies remain to be explored. Stay tuned!

_All illustrations are original content and should be credited accordingly._

<div>&copy; <b>Debucean Caius-Ioan</b> @ github.com/caiusdebucean</div>
<div> <i>Created in the first half of 2020 during COVID-19 social distancing regime by Caius-Ioan Debucean </i></div>