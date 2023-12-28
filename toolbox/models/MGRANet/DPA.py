from skimage.morphology import binary_opening, binary_closing, disk, binary_erosion, binary_dilation
from skimage.util import invert
from skimage.color import rgb2gray
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
#
im = rgb2gray(imread('./depth.png'))
print(np.max(im))
im[im <= 0.5] = 0
im[im > 0.5] = 1
plt.gray()
plt.figure(figsize=(20,10))
plt.subplot(231)
plt.imshow(im)
plt.title('original', size=20)
plt.axis('off')
plt.subplot(2,3,2)

###DPA###

# im1 = invert(binary_closing(invert(im), disk(6)))
im1 = binary_closing(im, disk(6))
plt.imshow(im1)
plt.title('closing with disk size ' + str(6), size=20)
plt.axis('off')
plt.subplot(2,3,5)
im1 = binary_erosion(im, disk(12))
plt.imshow(im1)
plt.title('erosion with disk size ' + str(12), size=20)
plt.axis('off')
plt.subplot(2,3,6)
im1 = binary_dilation(im, disk(6))
plt.imshow(im1)
plt.title('dilation with disk size ' + str(6), size=20)
plt.axis('off')
plt.show()
