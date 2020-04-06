In this folder I will keep track of my code modifications.

**Making the model work with the updated libraries**
1. 'Pix2Vox/utils/binvox_visualization.py' - Commented line 19 as matplotlib doesn't allow it anymore
2. 'Pix2Vox/core/test.py' - Added output directory and TensorboardX write compatibility for image results from net and gt - line 39 
3.  _OPTIONAL_ : 'Pix2Vox/core/test.py' - Removed the 3 samples images output limit - line 155
4. 'Pix2Vox/core/test.py' - Transposed rendering_views variable twice, as the supported type for TensorboardX is (C x W x H) and the one present was (W x H x C) - line 161
5. 'Pix2Vox/utils/binvox_visualization.py' - Added idx parameter to function to pass it to the save function for every test image, as the previous save function was overwriting output photos - line 22

**Installed New Packages**

*   **echoAI** - package containing a vast number of activation functions (including _Mish_, _Swish_, etc.)

**Experimentation**
1. 'config.py' - added *NETWORK.ALTERNATIVE_ACTIVATION_(A/B)* which is a function to help implement different activation functions in _models_ files (architecture modules)
2. 'decoder.py' 'encoder.py' 'merger.py' refiner.py' - added support for configurable activation functions from 'config.py' file.
3. **NEW DIRECTORY** - Added _Optimizers_ folder, containing _RAdam, Lookahead, Ranger_. 
4. '/core/train.py' - added optimizer options, which are chosen in 'config.py' without modifying the code