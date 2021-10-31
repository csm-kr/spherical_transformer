import math
import numpy as np
from utils.rotation_util import calculate_Rmatrix_from_phi_theta, rotate_map_given_R,\
    rotate_map_given_phi_theta_efficient, cartesian_to_spherical
from tqdm import tqdm

def make_quasi_equidistant_points(samples=1):

    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))
    points = np.vstack(points)
    return points

def compute_LUT(num_R, H, W):
    points = make_quasi_equidistant_points(num_R)
    R_list = []
    print('building rotation matrices')
    for i in tqdm(range(points.shape[0])):
        point = points[i, :]
        [phi, theta] = cartesian_to_spherical(point[0], point[1], point[2])
        [map_x, map_y] = rotate_map_given_phi_theta_efficient(phi, theta, H, W)
        R_list.append({'map_x':map_x, 'map_y':map_y})
    return R_list

def build_rotation_for_PanoVal(p_list, t_list, H, W):
    R_list = []
    for phi, theta in zip(p_list, t_list):
        [map_x, map_y] = rotate_map_given_phi_theta_efficient(phi, theta, H, W)
        R_list.append({'map_x':map_x, 'map_y':map_y, 'phi':phi, 'theta':theta})
    return R_list
