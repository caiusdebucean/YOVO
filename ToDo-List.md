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
* Research posibility to add improvements by generating depth maps with [MonoDepthNN](https://github.com/intel-isl/MiDaS/blob/master/run.py) from the [3D Photo Inpaiting](https://github.com/vt-vl-lab/3d-photo-inpainting)


**Specific things *to do* in the future:**

- [ ] Try **RangerLARS** optimizer
- [ ] Implement  backbone alternative to MobileNetV2 (HR-Net or U-net, possibly Darknet-19)
- [ ] Explore the other [YOLOV4](https://arxiv.org/pdf/2004.10934v1.pdf) improvements techniques 
- [ ] Try [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/pdf/1901.05555.pdf)
- [ ] Add attention modules
- [ ] Revamp Refiner
- [ ] _(OPTIONAL):_ Try _K3D_ package, for visualization (feasability test)

**Things which I have *done*:**

- [x] Correct original idea and make the code *Work*
- [x] Add and test **Ranger** optimizer (along with _RAdam_ and _Lookahead_)
- [x] Add and test the **Mish** activation function (with crude all-throughout replacement)
- [x] Code adaptive save mechanism to isolate training instances more easily
- [x] Add 3D interactive representation using [kaolin](https://github.com/NVIDIAGameWorks/kaolin) 
- [x] Implement beta-MobileNetV2 architecture
- [x] Learn to use **TensorboardX**
- [x] Find a **good combination** between variations of _relu_ and _Mish/ELU_
- [x] **REPAIR** and improve the function activation alternatives
- [x] Introduce **Multi-level** feature extraction and reconstruction
- [x] Better understand and tweak **Ranger** optimizer (_-> RMSProp -> Momentum -> Adam -> RAdam -> Lookahead_)
- [x] Add regularization method **DropBlock**
- [x] **Beat SotA** (Pix2Vox)
- [x] Add heatmap visualization to volumes
- [x] Add configurations for multi-level volume visualization
- [x] Add configurations for 360 gif of volumes and save input image
- [x] Implement modifications and introduce YOVO-s and YOVO-e
