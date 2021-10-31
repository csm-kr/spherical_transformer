from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R,\
    rotate_map_given_phi_theta_efficient, cartesian_to_spherical
from cupyx.scipy.ndimage import map_coordinates
import cv2
from scipy.ndimage import map_coordinates as mc
import numpy as np

img = cv2.imread('E:\\Downloads\\SUN360_panoramas_1024x512\\pano1024x512/indoor/bedroom/pano_aaacisrhqnnvoq.jpg')
[map_x, map_y] = rotate_map_given_phi_theta_efficient(phi=120, theta=21, height=512, width=1024)
# inds = [map_x, map_y]
# inds = [[0,0], [0,0]]
# mc(img, inds, order=1)
# import numpy as np

# import numpy
import cupy
# data = numpy.array([[4, 1, 3, 2],
#                                [7, 6, 8, 5],
#                                [3, 5, 3, 6]], order='F')
# idx = numpy.indices(data.shape) - 1
# print(data.shape, idx.shape)
# out = mc(data, idx)
# print(out)
import time
def cupy_remap(img, map_x, map_y, order=5):
    # to cuda
    img = cupy.asarray(img)
    maps = np.stack([map_y, map_x])
    # map_x = cupy.asarray(map_x)
    # map_y = cupy.asarray(map_y)
    maps = cupy.asarray(maps)
    t = 0
    for i in range(1000):
        tic = time.time()
        r = map_coordinates(img[:,:,0], maps, order=order)
        g = map_coordinates(img[:,:,1], maps, order=order)
        b = map_coordinates(img[:,:,2], maps, order=order)
        toc = time.time()
        img = cupy.stack([r,g,b],axis=2)
        t += (toc-tic)
    return cupy.asnumpy(img), t


if __name__ == '__main__':
    img, t = cupy_remap(img, map_x, map_y)
    print(t)