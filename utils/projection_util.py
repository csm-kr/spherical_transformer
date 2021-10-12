import math
import numpy as np
from utils.rotation_util import spherical_to_cartesian

# refer to s2cnn code : https://github.com/jonkhler/s2cnn/blob/master/examples/mnist/gendata.py


def get_projection_grid(b, grid_type="Driscoll-Healy"):
    ''' returns the spherical grid in euclidean
    coordinates, where the sphere's center is moved
    to (0, 0, 1)'''
    theta, phi = meshgrid(b=b, grid_type=grid_type)
    x_ = np.sin(theta) * np.cos(phi)
    y_ = np.sin(theta) * np.sin(phi)
    z_ = np.cos(theta)
    return x_, y_, z_



def meshgrid(b, grid_type='Driscoll-Healy'):
    """
    Create a coordinate grid for the 2-sphere.
    There are various ways to setup a grid on the sphere.

    if grid_type == 'Driscoll-Healy', we follow the grid_type from [4], which is also used in [5]:
    beta_j = pi j / (2 b)     for j = 0, ..., 2b - 1
    alpha_k = pi k / b           for k = 0, ..., 2b - 1

    if grid_type == 'ERP', we follow equi-rectangular projection:
    beta_j = pi j / (2 b)     for j = 0, ..., b
    alpha_k = pi k / b           for k = 0, ..., 2b - 1

    if grid_type == 'SOFT', we follow the grid_type from [1] and [6]
    beta_j = pi (2 j + 1) / (4 b)   for j = 0, ..., 2b - 1
    alpha_k = pi k / b                for k = 0, ..., 2b - 1

    if grid_type == 'Clenshaw-Curtis', we use the Clenshaw-Curtis grid, as defined in [2] (section 6):
    beta_j = j pi / (2b)     for j = 0, ..., 2b
    alpha_k = k pi / (b + 1)    for k = 0, ..., 2b + 1

    if grid_type == 'Gauss-Legendre', we use the Gauss-Legendre grid, as defined in [2] (section 6) and [7] (eq. 2):
    beta_j = the Gauss-Legendre nodes    for j = 0, ..., b
    alpha_k = k pi / (b + 1),               for k = 0, ..., 2 b + 1

    if grid_type == 'HEALPix', we use the HEALPix grid, see [2] (section 6):
    TODO

    if grid_type == 'equidistribution', we use the equidistribution grid, as defined in [2] (section 6):
    TODO

    [1] SOFT: SO(3) Fourier Transforms
    Kostelec, Peter J & Rockmore, Daniel N.

    [2] Fast evaluation of quadrature formulae on the sphere
    Jens Keiner, Daniel Potts

    [3] A Fast Algorithm for Spherical Grid Rotations and its Application to Singular Quadrature
    Zydrunas Gimbutas Shravan Veerapaneni

    [4] Computing Fourier transforms and convolutions on the 2-sphere
    Driscoll, JR & Healy, DM

    [5] Engineering Applications of Noncommutative Harmonic Analysis
    Chrikjian, G.S. & Kyatkin, A.B.

    [6] FFTs for the 2-Sphere – Improvements and Variations
    Healy, D., Rockmore, D., Kostelec, P., Moore, S

    [7] A Fast Algorithm for Spherical Grid Rotations and its Application to Singular Quadrature
    Zydrunas Gimbutas, Shravan Veerapaneni

    :param b: the bandwidth / resolution
    :return: a meshgrid on S^2
    """
    return np.meshgrid(*linspace(b, grid_type), indexing='ij')


def linspace(b, grid_type='Driscoll-Healy'):
    if grid_type == 'Driscoll-Healy':
        beta = np.arange(2 * b) * np.pi / (2. * b)
        alpha = np.arange(2 * b) * np.pi / b
    if grid_type == 'ERP':
        beta = np.arange(b) * np.pi / b
        alpha = np.arange(2 * b) * np.pi / b
    return beta, alpha


def rotate_grid(rot, grid):
    x, y, z = grid
    xyz = np.array((x, y, z))
    x_r, y_r, z_r = np.einsum('ij,jab->iab', rot, xyz)
    return x_r, y_r, z_r


def cartesian_to_spherical(x, y, z):
    # Input:
    #  x,y,z
    # Output:
    #  x,y,z (cartesian) that corresponds to rho,phi,theta
    # Goal:
    # convert spherical coordinates to cartesian coordinate
    # About convention :
    #  phi = arccos(z/r) , theta = arctan(y/x) range of phi [0,pi], range of theta [0,2pi]
    x_2 = math.pow(x, 2)
    y_2 = math.pow(y, 2)
    z_2 = math.pow(z, 2)

    theta = float(math.atan2(y, x))
    # atan2 returns value of which range is [-pi,pi], range of theta is [0,2pi] so if theta is negative value,actual value is theta+2pi
    if theta < 0:
        theta = theta + 2 * math.pi

    # theta = theta % (2* PI) # potential ERROR : phi [0,pi] theta [0,2pi] but atan2 returns value [-pi,pi]

    rho = x_2 + y_2 + z_2
    rho = math.sqrt(rho)
    phi = math.acos(z / rho)
    phi = np.rad2deg(phi)
    theta = np.rad2deg(theta)

    return [phi, theta]


def spherical_to_plane(phi, theta, h, w):
    return phi/180*h, theta/360*w


def plane_to_spherical(h, w, y, x):
    return np.rad2deg(y/h*np.pi), np.rad2deg(x/w*2*np.pi)


def calculate_2d_rotate_matrix(theta):
    rot = [[np.cos(theta), -np.sin(theta), 0],
           [np.sin(theta), np.cos(theta), 0],
           [0, 0, 1]]
    rot = np.array(rot)
    return rot


def projection(point:tuple, target_var, target_value):
    # project point to plane
    if target_var not in ['x', 'y', 'z']:
        raise RuntimeError('target var must be x or y or z')

    if target_var == 'x':
        target_var = point[0]
    elif target_var == 'y':
        target_var = point[1]
    elif target_var == 'z':
        target_var = point[2]

    eps = 1e-4
    #if abs(target_var) < eps:
    #    raise RuntimeError(f'target value is too small. It is smaller than {eps}')

    ratio = target_value/(target_var)
    point_target = tuple(map(lambda x: x*ratio, point))
    return point_target


def make_fov_projection_map(flat_img_h, flat_img_w, phi_fov, theta_fov):

    # this gives you a remap matrix_x and remap_matrix_y for projecting a flat image on the sphere surface
    # you need to specify resolution of flat image and phi fov, theta fov to specify on which region the image is projected.
    # define resolution of omni directional image according to the FOV

    omni_h = int((180 / phi_fov) * flat_img_h)
    omni_w = int((180 / theta_fov) * flat_img_w)

    phi_up = 90 - phi_fov / 2
    phi_down = 90 + phi_fov / 2
    theta_right = +theta_fov / 2
    theta_left = -theta_fov / 2
    right_up = spherical_to_cartesian(phi_up, theta_right)
    left_up = spherical_to_cartesian(phi_up, theta_left)
    right_down = spherical_to_cartesian(phi_down, theta_right)
    left_down = spherical_to_cartesian(phi_down, theta_left)

    right_up_f = projection(right_up, 'x', 1)
    left_up_f = projection(left_up, 'x', 1)
    right_down_f = projection(right_down, 'x', 1)
    left_down_f = projection(left_down, 'x', 1)

    y_dimension_f = right_down_f[1] - left_down_f[1]
    z_dimension_f = right_up_f[2] - right_down_f[2]

    remap_matrix_x = np.zeros((omni_h, omni_w), dtype=np.float32)
    remap_matrix_y = np.zeros((omni_h, omni_w), dtype=np.float32)
    eps = 1e-7
    for y in range(0, omni_h):
        for x in range(0, omni_w):
            cur_phi, cur_theta = plane_to_spherical(omni_h, omni_w, y, x) # cur phi [90-fov/2, 90+fov/2] cur theta [180-fov/2, 180+fov/2]
            p_on_sphere = spherical_to_cartesian(cur_phi, cur_theta)

            if p_on_sphere[0] >= 0:  # collide to x = 1 in a reverse direction
                remap_matrix_x[y, x] = -1
                remap_matrix_y[y, x] = -1
                continue

            if -eps <= p_on_sphere[0] <= eps:  # doesnt collide to x = 1 plane
                remap_matrix_x[y, x] = -1
                remap_matrix_y[y, x] = -1
                continue

            p_on_plane = projection(p_on_sphere, 'x', 1)
            y_on_plane = p_on_plane[1]
            z_on_plane = p_on_plane[2]

            y_condition = left_down_f[1] <= p_on_plane[1] <= right_down_f[1]
            z_condition = left_down_f[2] <= p_on_plane[2] <= left_up_f[2]
            if z_condition and y_condition:
                offset_y = y_on_plane - left_up_f[1]
                offset_z = z_on_plane - left_up_f[2]
                scaled_y = offset_y / y_dimension_f  # has to be from 0 to 1
                scaled_z = -offset_z / z_dimension_f  # has to be from 0 to 1
                p_on_flat_x = scaled_y * flat_img_w
                p_on_flat_y = scaled_z * flat_img_h
                remap_matrix_x[omni_h-y, x] = p_on_flat_x
                remap_matrix_y[omni_h-y, x] = p_on_flat_y
            else:
                remap_matrix_x[y, x] = -1
                remap_matrix_y[y, x] = -1

    return remap_matrix_x, remap_matrix_y
