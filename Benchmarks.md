This is a folder dedicated to benchmarking the different methods I use in improving the existing _Pix2Vox_.

**SoTA Pix2Vox-A Fully trained**

![Results](https://i.imgur.com/pwHlSiU.png)

___
___
___

For the sake of time saving, I will try comparing the following methods after **only 20 epochs**, so as to avoid catastrophic failures for intensive trainings.

**Original Pix2Vox-A**

*   average _batch time:_ 0.6s

![Results](https://i.imgur.com/FaW8XbJ.png)

___

**Pix2Vox-A with MobileNet-V2 pretrained Encoder, Mish activations and Ranger optimizer**

_**This particular architecture breaks the current overall SotA held by the original Pix2Vox**_
_I trained this at 32 batch size, at average batch time = 0.37s_
![Results](https://i.imgur.com/zATbeXO.png)
___

**Pix2Vox-A with Ranger optimizer**

*   average _batch time:_ 0.6s

![Results](https://i.imgur.com/0RbMwLp.png)

_Observation:_ This is a crud implementation as the Ranger optimizer didn't have the best hyperparametrization. Modification to the _Lookahead_ to _k=5_ instead of 6, may prove problematic when benchmarking divisible epochs.
___

**Pix2Vox-A with Ranger optimizer and Mish activation**

*   average _batch time:_ 0.7s
*   This proves to be the best method **so far!**

![Results](https://i.imgur.com/MOumhCJ.png)

_Observation:_ This is a crude attempt. Normally, Pix2Vox uses different variations of relu/elu throughout the _core models_. This was an attempt at replacing every of those activations with _Mish_. Bigger training time difference is also a factor that will need further investigation, as the _batch time_ is increasing as training progresses.