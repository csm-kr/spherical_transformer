import numpy as np
# for erp
from utils.projection_util import get_projection_grid

# for hd
from hausdorff_distance import hausdorff_dist

# sampling
from samplings.cube_sampling import inflate_cube
from samplings.icosahedron_sampling import inflate_icosahedron


def rand_uniform_hypersphere(N, p):
    if (p <= 0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")
    # Check N>0 and is an int
    if (N <= 0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")
    v = np.random.normal(0, 1, (N, p))
    v = np.divide(v, np.linalg.norm(v, axis=1, keepdims=True))
    return v


# sampling uniformly from *
num_samples = 100000
iterate = False

v = rand_uniform_hypersphere(N=num_samples, p=3)

# -------------------- ERP --------------------
'''
bandwidth : 25 -> 1250
bandwidth : 50 -> 5000
bandwidth : 100 -> 20000
'''
bandwidth = 100
points = get_projection_grid(b=bandwidth, grid_type='ERP')
u = np.stack([points[0].reshape(-1), points[1].reshape(-1), points[2].reshape(-1)], axis=1)
print("num_samples :", u.shape)
print("num_uniform points :", v.shape)
hd_dist = hausdorff_dist(u, v)
print('ERP Hausdorff Dist:', hd_dist)

if iterate:
    for i in range(10):
        v = rand_uniform_hypersphere(N=num_samples, p=3)
        hd_dist = hausdorff_dist(u, v)
        print(hd_dist)

# -------------------- CUBE --------------------
'''
num_edge : 14 -> 6 * 15 * 15 = 1350
num_edge : 29 -> 6 * 29 * 29 = 5046 
num_edge : 59 -> 6 * 58 * 58 = 20184 
'''
vertex_list = np.array(inflate_cube(num_edge=58)).reshape(-1, 3)               # [1280, 3]
u = vertex_list
print("num_samples :", u.shape)
print("num_uniform points :", v.shape)
hd_dist = hausdorff_dist(u, v)
print('Cube Hausdorff Dist:', hd_dist)

if iterate:
    for i in range(10):
        v = rand_uniform_hypersphere(N=num_samples, p=3)
        hd_dist = hausdorff_dist(u, v)
        print(hd_dist)

# -------------------- ICOSAHEDRON --------------------
'''
division_level = 3 -> 20 * 4 ** 3 = 1280
division_level = 4 -> 20 * 4 ** 4 = 5120
division_level = 5 -> 20 * 4 ** 5 = 20480
'''
vertex_list = np.array(inflate_icosahedron(division_level=5)).reshape(-1, 3)  # [1280, 3]
u = vertex_list
print("num_samples :", u.shape)
print("num_uniform points :", v.shape)
hd_dist = hausdorff_dist(u, v)
print('ICOSA Hausdorff Dist:', hd_dist)

if iterate:
    for i in range(10):
        v = rand_uniform_hypersphere(N=num_samples, p=3)
        hd_dist = hausdorff_dist(u, v)
        print(hd_dist)
