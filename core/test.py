# -*- coding: utf-8 -*-

import json
import time
import numpy as np
import os
import torch
import torch.backends.cudnn
import torch.utils.data
import kaolin as kal
import utils.binvox_visualization
import utils.data_loaders
import utils.data_transforms
import utils.network_utils
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from datetime import datetime as dt

from models.encoder import Encoder
from models.decoder import Decoder
from models.refiner import Refiner
from models.merger import Merger


def test_net(cfg,
             epoch_idx=-1,
             output_dir=None,
             test_data_loader=None,
             test_writer=None,
             encoder=None,
             decoder=None,
             refiner=None,
             merger=None):
    # Enable the inbuilt cudnn auto-tuner to find the best algorithm to use
    torch.backends.cudnn.benchmark = True

    #added image outputs
    if output_dir is None:
        save_path = cfg.DIR.OUT_PATH + '/' + cfg.DIR.NAME
        orig_save_path = save_path
        save_index = 0
        save_len = len(save_path)
        while os.path.exists(save_path):
            save_path = orig_save_path
            save_path = save_path + '_V' + str(save_index)
            save_index += 1
        output_dir = os.path.join(save_path, '%s', dt.now().isoformat())
    log_dir = output_dir % 'logs'
    test_writer = SummaryWriter(os.path.join(log_dir, 'test'))
    #also TensorboardX writer

    # Load taxonomies of dataset
    taxonomies = []
    with open(cfg.DATASETS[cfg.DATASET.TEST_DATASET.upper()].TAXONOMY_FILE_PATH, encoding='utf-8') as file:
        taxonomies = json.loads(file.read())
    taxonomies = {t['taxonomy_id']: t for t in taxonomies}

    # Set up data loader
    if test_data_loader is None:
        # Set up data augmentation
        IMG_SIZE = cfg.CONST.IMG_H, cfg.CONST.IMG_W
        CROP_SIZE = cfg.CONST.CROP_IMG_H, cfg.CONST.CROP_IMG_W
        test_transforms = utils.data_transforms.Compose([
            utils.data_transforms.CenterCrop(IMG_SIZE, CROP_SIZE),
            utils.data_transforms.RandomBackground(cfg.TEST.RANDOM_BG_COLOR_RANGE),
            utils.data_transforms.Normalize(mean=cfg.DATASET.MEAN, std=cfg.DATASET.STD),
            utils.data_transforms.ToTensor(),
        ])

        dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING[cfg.DATASET.TEST_DATASET](cfg)
        test_data_loader = torch.utils.data.DataLoader(dataset=dataset_loader.get_dataset(
            utils.data_loaders.DatasetType.TEST, cfg.CONST.N_VIEWS_RENDERING, test_transforms),
                                                       batch_size=1,
                                                       num_workers=1,
                                                       pin_memory=True,
                                                       shuffle=False)

    # Set up networks
    if decoder is None or encoder is None:
        encoder = Encoder(cfg)
        decoder = Decoder(cfg)
        refiner = Refiner(cfg)
        merger = Merger(cfg)

        if torch.cuda.is_available():
            encoder = torch.nn.DataParallel(encoder).cuda()
            decoder = torch.nn.DataParallel(decoder).cuda()
            refiner = torch.nn.DataParallel(refiner).cuda()
            merger = torch.nn.DataParallel(merger).cuda()

        print('[INFO] %s Loading weights from %s ...' % (dt.now(), cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        epoch_idx = checkpoint['epoch_idx']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if cfg.NETWORK.USE_REFINER:
            refiner.load_state_dict(checkpoint['refiner_state_dict'])
        if cfg.NETWORK.USE_MERGER:
            merger.load_state_dict(checkpoint['merger_state_dict'])

    # Set up loss functions
    bce_loss = torch.nn.BCELoss()

    # Testing loop
    n_samples = len(test_data_loader)
    test_iou = dict()
    encoder_losses = utils.network_utils.AverageMeter()
    refiner_losses = utils.network_utils.AverageMeter()

    # Switch models to evaluation mode
    encoder.eval()
    decoder.eval()
    refiner.eval()
    merger.eval()

    rend_exist = False #variable for not creating more rendering folders
    last_taxonomy = None
    current_taxonomy = None
    taxonomy_render_count = 0
    
    for sample_idx, (taxonomy_id, sample_name, rendering_images, ground_truth_volume) in enumerate(test_data_loader):
        start_time = time.time()
        taxonomy_id = taxonomy_id[0] if isinstance(taxonomy_id[0], str) else taxonomy_id[0].item()
        sample_name = sample_name[0]

        current_taxonomy = taxonomy_id
        with torch.no_grad():
            # Get data from data loader
            rendering_images = utils.network_utils.var_or_cuda(rendering_images)
            ground_truth_volume = utils.network_utils.var_or_cuda(ground_truth_volume)

            # Test the encoder, decoder, refiner and merger
            image_features = encoder(rendering_images)
            raw_features, generated_volume = decoder(image_features)

            if cfg.TEST.GENERATE_MULTILEVEL_VOLUMES:
                decoder_volume = torch.clone(generated_volume)
                decoder_features = torch.clone(raw_features)


            metric_autoencoder = 0
            if cfg.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
                generated_volume = merger(raw_features, generated_volume)
                if cfg.TEST.GENERATE_SIMPLE_VOLUME:
                    autoencoder_volume = torch.clone(generated_volume).detach()
                    _volume = torch.ge(autoencoder_volume, 0.4).float()
                    intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                    union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                    metric_autoencoder = (intersection / union).item()
            else:
                generated_volume = torch.mean(generated_volume, dim=1)
                
            encoder_loss = bce_loss(generated_volume, ground_truth_volume) * 10

            if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
                generated_volume = refiner(generated_volume)
                refiner_loss = bce_loss(generated_volume, ground_truth_volume) * 10
            else:
                refiner_loss = encoder_loss
            end_time = time.time()
            total_time = end_time - start_time
            # Append loss and accuracy to average metrics
            encoder_losses.update(encoder_loss.item())
            refiner_losses.update(refiner_loss.item())

            # IoU per sample
            sample_iou = []
            best_refined_volume = 0
            for th in cfg.TEST.VOXEL_THRESH:
                _volume = torch.ge(generated_volume, th).float()
                intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                current_metric = (intersection / union).item()
                if current_metric > best_refined_volume:
                    best_refined_volume = current_metric
                sample_iou.append(current_metric)

            # IoU per taxonomy
            if taxonomy_id not in test_iou:
                test_iou[taxonomy_id] = {'n_samples': 0, 'iou': []}
            test_iou[taxonomy_id]['n_samples'] += 1
            test_iou[taxonomy_id]['iou'].append(sample_iou)

            is_good_sample = False
            # print(sample_iou)
            for spl_iou in sample_iou:
                # print("one instance has ",spl_iou)
                if spl_iou > cfg.TEST.RENDER_THRESHOLD:
                    is_good_sample = True
            print(sample_idx)
            # Append generated volumes to TensorBoard
            # if output_dir and sample_idx < cfg.TEST.N_VIEW: #Only prints 3 images - remove second condition for all dataset
            if current_taxonomy != last_taxonomy:
                last_taxonomy = current_taxonomy
                taxonomy_render_count = 0
            if output_dir and is_good_sample and taxonomy_render_count < cfg.TEST.NO_OF_RENDERS and best_refined_volume > cfg.TEST.RENDER_THRESHOLD and (best_refined_volume - metric_autoencoder) > cfg.TEST.DIFFERENCE_THESHOLD:# and current_taxonomy=='03001627':
                if current_taxonomy == last_taxonomy:
                    taxonomy_render_count+=1
                print("Found a good sample")
                #Decoder print 3 differences
                if cfg.TEST.GENERATE_MULTILEVEL_VOLUMES:
                    if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS is not None:
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "plane":
                            volume_class = '02691156'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "bench":
                            volume_class = '02828884'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "cabinet":
                            volume_class = '02933112'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "car":
                            volume_class = '02958343'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "chair":
                            volume_class = '03001627'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "display":
                            volume_class = '03211117'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "lamp":
                            volume_class = '03636649'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "speaker":
                            volume_class = '03691459'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "rifle":
                            volume_class = '04090263'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "sofa":
                            volume_class = '04256520'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "table":
                            volume_class = '04379243'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "phone":
                            volume_class = '04401088'
                        if cfg.TEST.CLASS_TO_GENERATE_MULTI_LEVELS == "boat":
                            volume_class = '04530566'
                    else:
                        print("Please specify which object class to generate multi-level volumes! No class chosen.")
                        volume_class = False
                        # exit()

                        
                    merge_total_volume = merger(decoder_features, decoder_volume)
                    decoder_volume = decoder_volume.detach().squeeze()
                    decoder_features = decoder_features.detach().squeeze()
                    print("Decoder volumes:/n")
                    for i in range(3):
                        for th in cfg.TEST.VOXEL_THRESH:
                            print(f"Decoder volume number {i} with threshold {th} :")
                            _volume = torch.ge(decoder_volume[i], th).float()
                            intersection = torch.sum(_volume.mul(ground_truth_volume)).float()
                            union = torch.sum(torch.ge(_volume.add(ground_truth_volume), 1)).float()
                            metric = (intersection / union).item()
                            print(metric)
                            if th == 0.4:
                                gv = decoder_volume[i].cpu().numpy()
                                img_dir = output_dir % 'Decoder_volumes'
                                rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'Multi_level_views'), epoch_idx, sample_idx+i, save_gif=False, color_map="bone", interactive_show = True)
                        print('/n')
                    gv = merge_total_volume.cpu().numpy()
                    rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'Multi_level_views'), epoch_idx, sample_idx+4, save_gif=False, color_map="Spectral")
                #Decoder print 3 differences
                img_dir = output_dir % 'images_from_test'
                renderer_dir = output_dir % 'renderer'
                # Volume Visualization
                gv = generated_volume.cpu().numpy()
                if cfg.TEST.VIEW_KAOLIN == True:
                    kaolin_gv = np.copy(gv)
                    kaolin_gv = np.squeeze(kaolin_gv,axis=0)
                    kal.visualize.show(kaolin_gv, mode='voxels')
                rendering_views = utils.binvox_visualization.get_volume_views(gv, os.path.join(img_dir, 'test'), epoch_idx, sample_idx, save_gif=cfg.TEST.SAVE_GIF, color_map="viridis", interactive_show = True)
                if cfg.TEST.GENERATE_SIMPLE_VOLUME :
                    if best_refined_volume - metric_autoencoder > cfg.TEST.DIFFERENCE_THESHOLD:
                        autoencoder_volume = autoencoder_volume.cpu().numpy()
                        rendering_autoencoder = utils.binvox_visualization.get_volume_views(autoencoder_volume, os.path.join(img_dir, 'autoencoder_photos'), epoch_idx, sample_idx, save_gif=cfg.TEST.SAVE_GIF, color_map="viridis")
                rendering_views = np.transpose(rendering_views,(2,0,1))

                #Supported type is (C x W x H) and current one is (W x H x C)         
                test_writer.add_image('Test Sample#%02d/Volume Reconstructed' % sample_idx, rendering_views, epoch_idx)
                _volume = torch.ge(ground_truth_volume, 0.001).float()
                gtv = _volume.cpu().numpy()
                if cfg.TEST.SAVE_RENDERED_IMAGE == True:
                    print(type(rendering_images))
                    rendering_images = torch.squeeze(rendering_images)
                    rendering_images = rendering_images.cpu()
                    print(rendering_images.size())
                    rendering_images = (rendering_images.permute(1, 2, 0))
                    
                    plt.imshow((rendering_images.numpy() * 255).astype(np.uint8))
                    if rend_exist == False:
                        os.mkdir(img_dir + '/rendering')
                        rend_exist = True
                    render_save = img_dir + '/rendering'
                    # render_save = renderer_dir
                    plt.savefig(render_save +'/test_'+ str(sample_idx) + "_" + str(taxonomy_id) + "_" + str(sample_name) + '.png', bbox_inches='tight', pad_inches=0)

                    print("Rendered an image")
                # if cfg.TEST.VIEW_KAOLIN == True:
                #     kaolin_gtv = np.copy(gtv)
                #     kaolin_gtv = np.squeeze(kaolin_gtv,axis=0)
                #     kal.visualize.show(kaolin_gtv, mode='voxels')
                
                rendering_views = utils.binvox_visualization.get_volume_views(gtv, os.path.join(img_dir, 'test_gt'), epoch_idx, sample_idx, test=True, save_gif=False)
                
                #Supported type is (C x W x H) and current one is (W x H x C)
                rendering_views = np.transpose(rendering_views,(2,0,1))
                test_writer.add_image('Test Sample#%02d/Volume GroundTruth' % sample_idx, rendering_views, epoch_idx)




            # Print sample loss and IoU
            print('[INFO] %s Test[%d/%d] Taxonomy = %s Sample = %s EDLoss = %.4f RLoss = %.4f IoU = %s' %
                  (dt.now(), sample_idx + 1, n_samples, taxonomy_id, sample_name, encoder_loss.item(),
                   refiner_loss.item(), ['%.4f' % si for si in sample_iou]))

    # Output testing results
    mean_iou = []
    for taxonomy_id in test_iou:
        test_iou[taxonomy_id]['iou'] = np.mean(test_iou[taxonomy_id]['iou'], axis=0)
        mean_iou.append(test_iou[taxonomy_id]['iou'] * test_iou[taxonomy_id]['n_samples'])
    mean_iou = np.sum(mean_iou, axis=0) / n_samples

    # Print Time statistics
    # print("Time statistics:")
    # avg_time = sum(total_time) / len(total_time)
    # print(f"Average time is {avg_time} seconds")

    # Print header
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    print('Baseline', end='\t')
    for th in cfg.TEST.VOXEL_THRESH:
        print('t=%.2f' % th, end='\t')
    print()
    # Print body
    for taxonomy_id in test_iou:
        print('%s' % taxonomies[taxonomy_id]['taxonomy_name'].ljust(8), end='\t')
        print('%d' % test_iou[taxonomy_id]['n_samples'], end='\t')
        if 'baseline' in taxonomies[taxonomy_id]:
            print('%.4f' % taxonomies[taxonomy_id]['baseline']['%d-view' % cfg.CONST.N_VIEWS_RENDERING], end='\t\t')
        else:
            print('N/a', end='\t\t')

        for ti in test_iou[taxonomy_id]['iou']:
            print('%.4f' % ti, end='\t')
        print()
    # Print mean IoU for each threshold
    print('Overall ', end='\t\t\t\t')
    # print(mean_iou)
    if cfg.DATASET.TEST_DATASET == 'ShapeNet':
        for mi in mean_iou:
            print('%.4f' % mi, end='\t')
        print('\n')

    # Add testing results to TensorBoard
    max_iou = np.max(mean_iou)
    if test_writer is not None:
        test_writer.add_scalar('EncoderDecoder/EpochLoss', encoder_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/EpochLoss', refiner_losses.avg, epoch_idx)
        test_writer.add_scalar('Refiner/IoU', max_iou, epoch_idx)

    return max_iou

