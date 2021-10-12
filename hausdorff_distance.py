import numpy as np
from scipy.spatial.distance import directed_hausdorff

# u = np.array([[1.0, 0.0, 0.5],
#               [0.0, 1.0, 2.2],
#               [-1.0, 0.0, 3.6],
#               [-0.9, 0.0, -1.0]])
# v = np.array([[2.0, 0.0, -2.2],
#               [2.1, 2.0, 2.0],
#               [-2.0, 0.0, 3.5],
#               [0.0, -4.0, 2.3]])
# a = directed_hausdorff(u, v)[0]
# b = directed_hausdorff(v, u)[0]
# hausdorff_dist = max(a, b)
# print(a, b)
# print('Hausdorff Dist:', hausdorff_dist)


def hausdorff_dist(u, v):
    a = directed_hausdorff(u, v)[0]
    b = directed_hausdorff(v, u)[0]
    ret = max(a, b)
    return ret


if __name__ == '__main__':
    u = np.array([[1.0, 0.0, 0.5],
                  [0.0, 1.0, 2.2],
                  [-1.0, 0.0, 3.6],
                  [-0.9, 0.0, -1.0]])
    v = np.array([[2.0, 0.0, -2.2],
                  [2.1, 2.0, 2.0],
                  [-2.0, 0.0, 3.5],
                  [0.0, -4.0, 2.3]])
    hd_dist = hausdorff_dist(u, v)
    print('Hausdorff Dist:', hd_dist)
