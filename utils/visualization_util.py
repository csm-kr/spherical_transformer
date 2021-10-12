import matplotlib.pyplot as plt
import numpy as np


def grid_2_points(grid):
    """
    tuple grid to points
    :param grid: tuple (x, y, z), each shape is (60, 60)
    :return: points (num_points, 3)
    """
    points = np.stack([grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)], axis=-1)  # N ^ 2, 3
    return points


def show_spheres(scale, points, rgb, label=[0, 0, 1]):
    """

    :param scale: int
    :param points: ndarray : (num_points, 3)
    :param rgb:
    :return:
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # axis scale setting
    ax.set_xlim3d(-1 * scale, 1 * scale)
    ax.set_ylim3d(-1 * scale, 1 * scale)
    ax.set_zlim3d(-0.8 * scale,  0.8 * scale)
    x, y, z = label

    # label
    ax.plot([0, scale * x], [0, scale * y], [0, scale * z])

    # how rotate they are
    phi2 = np.arctan2(y, x) * 180 / np.pi
    theta = np.arccos(z) * 180 / np.pi

    if phi2 < 0:
        phi2 = 360 + phi2

    # ax.set_aspect('equal')
    if rgb.shape != points.shape:
        rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
        rgb = np.concatenate([rgb, rgb, rgb], axis=1)

    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=1, facecolors=rgb, depthshade=False)  # data 색 입히기
    plt.legend(loc=2)

    # 90 도에서 보는 사진
    ax.view_init(-1 * theta, phi2)

    # 위에서 보는 사진
    # ax.view_init(-1 * theta + 90, phi2)

    plt.draw()
    plt.show()
