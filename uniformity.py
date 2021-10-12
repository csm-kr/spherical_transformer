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
v = rand_uniform_hypersphere(N=10000, p=3)

# -------------------- ERP --------------------
'''
bandwidth: 25 -> 1250
'''
bandwidth = 26
points = get_projection_grid(b=bandwidth, grid_type='ERP')
u = np.stack([points[0].reshape(-1), points[1].reshape(-1), points[2].reshape(-1)], axis=1)
print("num_samples :", u.shape)
print("num_uniform points :", v.shape)
hd_dist = hausdorff_dist(u, v)
print('ERP Hausdorff Dist:', hd_dist)

# -------------------- CUBE --------------------
'''
num_edge: 14 -> 6 * 14 * 14 = 1176
'''
vertex_list = np.array(inflate_cube(num_edge=15)).reshape(-1, 3)               # [1280, 3]
u = vertex_list
print("num_samples :", u.shape)
print("num_uniform points :", v.shape)
hd_dist = hausdorff_dist(u, v)
print('Cube Hausdorff Dist:', hd_dist)

# -------------------- ICOSAHEDRON --------------------
'''
division_level = 3 -> 20 * 4^3 = 1280
'''
vertex_list = np.array(inflate_icosahedron(division_level=3)).reshape(-1, 3)  # [1280, 3]
u = vertex_list
print("num_samples :", u.shape)
print("num_uniform points :", v.shape)
hd_dist = hausdorff_dist(u, v)
print('ICOSA Hausdorff Dist:', hd_dist)
