# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from easydict import EasyDict as edict

__C                                         = edict()
cfg                                         = __C

#
# Dataset Config
#
__C.DATASETS                                = edict()
__C.DATASETS.SHAPENET                       = edict()
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = './datasets/ShapeNet.json'
# __C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH  = './datasets/PascalShapeNet.json'

# __C.DATASETS.SHAPENET.RENDERING_PATH        = '/home/amuresan/Documents/CaiusD/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/media/caius/Elements/Licenta/Pix2VoxData/ShapeNetRendering/ShapeNetRendering/%s/%s/rendering/%02d.png'
# __C.DATASETS.SHAPENET.VOXEL_PATH            = '/home/amuresan/Documents/CaiusD/ShapeNetVox32/%s/%s/model.binvox'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/media/caius/Elements/Licenta/Pix2VoxData/ShapeNetVox32/ShapeNetVox32/%s/%s/model.binvox'

__C.DATASETS.PASCAL3D                       = edict()
__C.DATASETS.PASCAL3D.TAXONOMY_FILE_PATH    = './datasets/Pascal3D.json'
__C.DATASETS.PASCAL3D.ANNOTATION_PATH       = '/home/hzxie/Datasets/PASCAL3D/Annotations/%s_imagenet/%s.mat'
__C.DATASETS.PASCAL3D.RENDERING_PATH        = '/home/hzxie/Datasets/PASCAL3D/Images/%s_imagenet/%s.JPEG'
__C.DATASETS.PASCAL3D.VOXEL_PATH            = '/home/hzxie/Datasets/PASCAL3D/CAD/%s/%02d.binvox'
__C.DATASETS.PIX3D                          = edict()

# __C.DATASETS.PIX3D.TAXONOMY_FILE_PATH       = './datasets/Pix3D.json'
# __C.DATASETS.PIX3D.ANNOTATION_PATH          = '/media/caius/Elements/Licenta/Pix2VoxData/Pix3D/pix3d.json'
# __C.DATASETS.PIX3D.RENDERING_PATH           = '/media/caius/Elements/Licenta/Pix2VoxData/Pix3D/img/%s/%s.%s'
# __C.DATASETS.PIX3D.VOXEL_PATH               = '/media/caius/Elements/Licenta/Pix2VoxData/Pix3D/model/%s/%s/%s.binvox'
__C.DATASETS.PIX3D.TAXONOMY_FILE_PATH       = './datasets/customPix3D.json'
__C.DATASETS.PIX3D.ANNOTATION_PATH          = '/media/caius/Elements/Licenta/Pix2VoxData/CustomPix3D/pix3d.json'
__C.DATASETS.PIX3D.RENDERING_PATH           = '/media/caius/Elements/Licenta/Pix2VoxData/CustomPix3D/img/%s/%s.%s'
__C.DATASETS.PIX3D.VOXEL_PATH               = '/media/caius/Elements/Licenta/Pix2VoxData/CustomPix3D/model/%s/%s/%s.binvox'

#
# Dataset
#
__C.DATASET                                 = edict()
__C.DATASET.MEAN                            = [0.5, 0.5, 0.5]
__C.DATASET.STD                             = [0.5, 0.5, 0.5]
__C.DATASET.TRAIN_DATASET                   = 'ShapeNet'
__C.DATASET.TEST_DATASET                    = 'ShapeNet'
# __C.DATASET.TRAIN_DATASET                   = 'Pix3D'
# __C.DATASET.TEST_DATASET                    = 'Pix3D'
# __C.DATASET.TEST_DATASET                  = 'Pascal3D'
# __C.DATASET.TEST_DATASET                  = 'Pix3D'

#
# Common
#
__C.CONST                                   = edict()
__C.CONST.DEVICE                            = '0'
__C.CONST.RNG_SEED                          = 0
__C.CONST.IMG_W                             = 224       # Image width for input
__C.CONST.IMG_H                             = 224       # Image height for input
__C.CONST.N_VOX                             = 32
__C.CONST.BATCH_SIZE                        = 16
__C.CONST.N_VIEWS_RENDERING                 = 1        # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_W                        = 128       # Dummy property for Pascal 3D
__C.CONST.CROP_IMG_H                        = 128       # Dummy property for Pascal 3D
#
# Directories
#
__C.DIR                                     = edict()
__C.DIR.OUT_PATH                            = './output'
__C.DIR.RANDOM_BG_PATH                      = ''
__C.CONST.NAME                              = 'Test'  

#
# Network
#

__C.NETWORK                                 = edict()
__C.NETWORK.ENCODER                         = 'mobilenet' # What architecture to use: ['original', 'darknet19', 'mobilenet']
__C.NETWORK.DECODER                         = 'original' # What architecture to use: ['original']
__C.NETWORK.MERGER                          = 'original' # What architecture to use: ['original']
__C.NETWORK.REFINER                         = 'original' # What architecture to use: ['original']
__C.NETWORK.MULTI_LEVEL_TRAIN               = True
__C.NETWORK.LEAKY_VALUE                     = .2
__C.NETWORK.TCONV_USE_BIAS                  = False
__C.NETWORK.USE_REFINER                     = True
__C.NETWORK.USE_MERGER                      = True
__C.NETWORK.EXTEND_DECODER                  = False # Extends upsampling to avoid information loss
__C.NETWORK.EXTEND_REFINER                  = True # Requires refiner version
__C.NETWORK.REFINER_VERSION                 = 1 # [1 , 2] - Version 1 adds one abstraction level, version 2 adds 2 abstraction levels
__C.NETWORK.ALTERNATIVE_ACTIVATION_A        = 'mish' # ['relu', 'elu', 'leaky relu', 'mish']
__C.NETWORK.ALTERNATIVE_ACTIVATION_B        = 'elu' # ['relu', 'elu', 'leaky relu', 'mish']
__C.NETWORK.USE_DROPBLOCK                   = True
__C.NETWORK.DROPBLOCK_VALUE_2D              = 0.05 # what is the probability that a block is dropped
__C.NETWORK.DROPBLOCK_VALUE_3D              = 0.05 # what is the probability that a block is dropped

#
# Training
#
__C.TRAIN                                   = edict()
__C.TRAIN.RESUME_TRAIN                      = False
__C.TRAIN.NUM_WORKER                        = 4             # number of data workers
__C.TRAIN.NUM_EPOCHES                       = 250
__C.TRAIN.BRIGHTNESS                        = .4
__C.TRAIN.CONTRAST                          = .4
__C.TRAIN.SATURATION                        = .4
__C.TRAIN.NOISE_STD                         = .1
__C.TRAIN.RANDOM_BG_COLOR_RANGE             = [[225, 255], [225, 255], [225, 255]]
#__C.TRAIN.POLICY                            = 'adam'        # available options: sgd, adam, ranger
__C.TRAIN.POLICY                            = 'ranger'
__C.TRAIN.EPOCH_START_USE_REFINER           = 0
__C.TRAIN.EPOCH_START_USE_MERGER            = 0
__C.TRAIN.ENCODER_LEARNING_RATE             = 1e-3
__C.TRAIN.DECODER_LEARNING_RATE             = 1e-3
__C.TRAIN.REFINER_LEARNING_RATE             = 1e-3
__C.TRAIN.MERGER_LEARNING_RATE              = 1e-4
__C.TRAIN.ENCODER_LR_MILESTONES             = [150]
__C.TRAIN.DECODER_LR_MILESTONES             = [150]
__C.TRAIN.REFINER_LR_MILESTONES             = [150]
__C.TRAIN.MERGER_LR_MILESTONES              = [150]
__C.TRAIN.BETAS                             = (.9, .999)
__C.TRAIN.MOMENTUM                          = .9
__C.TRAIN.GAMMA                             = .5
__C.TRAIN.SAVE_FREQ                         = 10            # weights will be overwritten every save_freq epoch
__C.TRAIN.UPDATE_N_VIEWS_RENDERING          = False

#
# Testing options
#
__C.TEST                                    = edict()
__C.TEST.RANDOM_BG_COLOR_RANGE              = [[240, 240], [240, 240], [240, 240]]
__C.TEST.VOXEL_THRESH                       = [.2, .3, .4, .5]
__C.TEST.VIEW_KAOLIN                        = True  # Rendering during training with kaolin. This should be done locally, not through ssh
__C.TEST.N_VIEW                             = 1 # How many images should we save and render at test/validation time
__C.TEST.SAVE_RENDERED_IMAGE                = True # Save the input preprocessed image containing the object
__C.TEST.SAVE_GIF                           = False # Save GIF of 360 rotating volume
__C.TEST.NO_OF_RENDERS                      = 5 # How many examples to be saved for visualization
__C.TEST.RENDER_THRESHOLD                   = 0.85
__C.TEST.GENERATE_MULTILEVEL_VOLUMES        = True
__C.TEST.CLASS_TO_GENERATE_MULTI_LEVELS     = None #[None,"plane","bench","cabinet","car","chair","display","lamp","speaker","rifle","sofa","table","phone","boat"]
__C.TEST.GENERATE_SIMPLE_VOLUME             = False # Generate images from Merger, without refining
__C.TEST.DIFFERENCE_THESHOLD                = 0.2 # How much of a difference you want to have between final volume and refined volume. Between[0,1]. -1 to turn off
