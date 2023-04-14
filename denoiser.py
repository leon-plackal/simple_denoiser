import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.filters import unsharp_mask
from skimage import io
from scipy import ndimage as nd
import bm3d
from skimage.restoration import (denoise_tv_chambolle)

noisy = img_as_float(io.imread('images/veggies.jpg'))

# BM3D
BM3D_denoised = bm3d.bm3d(noisy, sigma_psd=0.17, stage_arg=bm3d.BM3DStages.ALL_STAGES)
# sharpening
result_3 = unsharp_mask(BM3D_denoised, radius=2, amount=2)
plt.imsave('images/veggies_sharpened.png', result_3, cmap='gray')

# denoise TV
# denoise_TV = denoise_tv_chambolle(noisy, weight=0.3)
# plt.imsave('images/TV_smoothed2.png', denoise_TV, cmap='gray')
#guassian denoise
# guassian_img = nd.gaussian_filter(noisy, sigma=3)
# plt.imsave('images/guass_smooth.png', guassian_img, cmap='gray')