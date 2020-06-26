# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import cv2
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.interpolation import rotate as R
import numpy as np
from matplotlib import animation

def get_volume_views(volume, save_dir, n_itr, idx, test=False, save_gif=False,color_map="bone"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # if test == False:
    #     np.save('test.npy',volume)
    
    #volume = np.load('../test.npy')

    volume = volume.squeeze()
    volume = R(volume, 90, axes = (1,2))
    volume = R(volume, -20, axes = (0,2))

    heatmap_volume = np.copy(volume)
    heatmap_volume = (heatmap_volume.squeeze())*255
    heatmap_volume = np.trunc(heatmap_volume)
    volume = volume.squeeze().__ge__(0.5)

    cmap = plt.get_cmap(color_map)
    norm= plt.Normalize(heatmap_volume.min(), heatmap_volume.max())

    fig = plt.figure()
    sub = fig.add_subplot(1,1,1,projection='3d')

    ax = fig.gca(projection=Axes3D.name)
    #MATPLOLTLIB doesn't allow anymore
    #ax.set_aspect('equal')

    def init():
        ax.scatter(volume[0],volume[1],volume[2], marker='o', s=20, c="goldenrod", alpha=0.6)
        return fig,

    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,

    if test == False:
        ax.voxels(volume, facecolors=cmap(norm(heatmap_volume)), edgecolor=None)
    else:
        ax.voxels(volume, facecolors='ivory', edgecolor='k', linewidths=0.4)

    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    #Added idx for avoiding overwriting output photos
    name = 'voxels_id_' + str(idx) + '_iter%06d.png' % n_itr

    gif_dir = os.path.join(save_dir, 'gifs')
    if not os.path.exists(gif_dir):
        os.mkdir(gif_dir)
    save_path = os.path.join(save_dir, name)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if save_gif == True:
        try:
            anim = animation.FuncAnimation(fig, animate,
                                    frames=360, interval=20, blit=True)
            # Save
            gif_name = 'voxels_id_' + str(idx) + '_iter%06d.gif' % n_itr
            gif_save_path = os.path.join(gif_dir, gif_name)
            anim.save(gif_save_path, fps=30, extra_args=['-vcodec', 'libx264'])
        except:
            plt.close()
            return cv2.imread(save_path)
    plt.close()
    return cv2.imread(save_path)

