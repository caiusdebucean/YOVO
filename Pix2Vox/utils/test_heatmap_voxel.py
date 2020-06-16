import cv2
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage.interpolation import rotate as R
import numpy as np
from matplotlib import animation

volume = np.load('../test.npy')
volume = volume.squeeze()
volume = R(volume, 90, axes = (1,2))
volume = R(volume, -20, axes = (0,2))


heatmap_volume = np.copy(volume)
heatmap_volume = (heatmap_volume.squeeze())*255
heatmap_volume = np.trunc(heatmap_volume)
volume = volume.squeeze().__ge__(0.5)

cmap = plt.get_cmap("viridis")
norm= plt.Normalize(heatmap_volume.min(), heatmap_volume.max())

fig = plt.figure()
sub = fig.add_subplot(1,1,1,projection='3d')

ax = fig.gca(projection=Axes3D.name)
#MATPLOLTLIB doesn't allow anymore
#ax.set_aspect('equal')
test = False


def init():
    ax.scatter(volume[0],volume[1].volume[2], marker='o', s=20, c="goldenrod", alpha=0.6)
    return fig,

def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

# Animate
ax.voxels(volume, facecolors=cmap(norm(heatmap_volume)), edgecolor=None)
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
plt.axis('off')

anim = animation.FuncAnimation(fig, animate,
                               frames=3, interval=20, blit=True)
# Save
anim.save('basic_animation.gif', fps=1, extra_args=['-vcodec', 'libx264'])
# if test == False:
#     ax.voxels(volume, facecolors=cmap(norm(heatmap_volume)), edgecolor=None)
# else:
#     ax.voxels(volume, facecolors='red', edgecolor='k')


#Added idx for avoiding overwriting output photos
# name = 'voxels_id_' + str(idx) + '_iter%06d.png' % n_itr
save_path = '/media/caius/Elements/Licenta/Bachelors-Degree-Project/Pix2Vox/utils/test_npy/test.png' #os.path.join(save_dir, name)
plt.show()
plt.savefig(save_path, bbox_inches='tight')
plt.close()