# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D


def get_volume_views(volume, save_dir, n_itr, idx):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    volume = volume.squeeze().__ge__(0.5)
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    #MATPLOLTLIB doesn't allow anymore
    #ax.set_aspect('equal')
    ax.voxels(volume,facecolors='red', edgecolor="k")
    #Added idx for avoiding overwriting output photos
    name = 'voxels_id_' + str(idx) + '_iter%06d.png' % n_itr
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return cv2.imread(save_path)