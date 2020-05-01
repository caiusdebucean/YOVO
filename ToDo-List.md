This is a file tracking the of improvements and modifications I want to add to the current state of Pix2Vox or other projects/topics I come across. 
_Minor changes, such as code modifications or small improvements will not be tracked in this document_

**General improvement *ideas*:**

* Try out different [optimizers](https://github.com/mgrankin/over9000)
* Try different backbone architecture (HR-Net, SqueezeNet, MobileNetV2, ResNet, etc.)
* Study what modules can be improved
* Try out and research personal super secret activation function
* Try out different augmentation and regularization methods
* Research photogrammetry (from DL research or apps like canvas.io or occipital projects)
* Research PapersWithCode for similar tasks
* Research posibility to add improvements by generating depth maps with ![MonoDepthNN](https://github.com/intel-isl/MiDaS/blob/master/run.py) from the [3D Photo Inpaiting](https://github.com/vt-vl-lab/3d-photo-inpainting)


**Specific things *to do* in the future:**

- [ ] **REPAIR** and improve the function activation alternatives
- [ ] Better understand and tweak **Ranger** optimizer (_-> RMSProp -> Momentum -> Adam -> RAdam -> Lookahead_)
- [ ] Try **RangerLARS** optimizer
- [ ] Implement HR-Net or U-net alternative for backbone (possibly Darknet-19)
- [ ] Add regularization method **DropBlock**
- [ ] Try out **Pix2Vox-F**
- [ ] Find a **good combination** between variations of _relu_ and _Mish/Swish_
- [ ] Learn to use **TensorboardX**
- [ ] _(OPTIONAL):_ Try _K3D_ package, for visualization (feasability test)

**Things which I have *done*:**

- [x] Correct Pix2Vox-A and made it *Work*
- [x] Add and tested **Ranger** optimizer (along with _RAdam_ and _Lookahead_)
- [x] Add and teste the **Mish** activation function (with crude all-throughout replacement)
- [x] Code adaptive save mechanism to isolate training instances more easily
- [x] Add 3D interactive representation using [kaolin](https://github.com/NVIDIAGameWorks/kaolin) 
- [x] Implement beta-MobileNetV2 architecture