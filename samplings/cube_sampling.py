import numpy as np
import matplotlib.pyplot as plt


def generate_cube():
    # generate vertices of cube
    # returns 8 vertices of cube
    V0 = np.array([1, 1, 1])
    V1 = np.array([1, -1, 1])
    V2 = np.array([-1, -1, 1])
    V3 = np.array([-1, 1, 1])
    V4 = np.array([1, 1, -1])
    V5 = np.array([1, -1, -1])
    V6 = np.array([-1, -1, -1])
    V7 = np.array([-1, 1, -1])
    return [V0, V1, V2, V3, V4, V5, V6, V7]


def generate_face_group():
    # from 8 vertices of cube
    # this generate 6 faces of icosahedron and then return the faces as a form of dictionary
    [v0, v1, v2, v3, v4, v5, v6, v7] = generate_cube()

    face_dict = dict()
    face_dict[1] = [v2, v3, v0, v1]  # top face
    face_dict[2] = [v0, v3, v7, v4]  # right face
    face_dict[3] = [v1, v0, v4, v5]  # front face
    face_dict[4] = [v2, v1, v5, v6]  # left face
    face_dict[5] = [v3, v2, v6, v7]  # back face
    face_dict[6] = [v7, v6, v5, v4]  # bottom face
    return face_dict


class Face:
    # this is a class for representing a face of icosahedron
    def __init__(self, v1, v2, v3, v4):
        self.v1 = v1  # left top
        self.v2 = v2  # right top
        self.v3 = v3  # left bottom
        self.v4 = v4  # right bottom

    def get_points(self, edge: int):
        """
        edge의 갯수를 받아서 한 face의 points 들을 리턴하는 함수
        :param edge: int
        :return: self.point_list: list
        """
        point_list = []
        # 모든축
        whole_dim = set([0, 1, 2])

        # 모든 v 가 같은 축을 찾자.
        eq_dim = np.argmax(np.abs(self.v1 + self.v2 + self.v3 + self.v4))
        remnant_dim = whole_dim - set([eq_dim])
        remnant_dim = np.sort(list(remnant_dim))

        if self.v1[remnant_dim[0]] - self.v2[remnant_dim[0]] != 0:
            v1v2dim = remnant_dim[0]
            v1v3dim = remnant_dim[1]
        else:
            v1v2dim = remnant_dim[1]
            v1v3dim = remnant_dim[0]

        if v1v2dim < v1v3dim:
            xs = np.linspace(self.v1[v1v2dim], self.v2[v1v2dim], edge)
            ys = np.linspace(self.v1[v1v3dim], self.v3[v1v3dim], edge)
        else:
            ys = np.linspace(self.v1[v1v2dim], self.v2[v1v2dim], edge)
            xs = np.linspace(self.v1[v1v3dim], self.v3[v1v3dim], edge)

        # mesh-grid 만들기
        xx, yy = np.meshgrid(xs, ys)
        points = np.concatenate([np.expand_dims(xx, axis=-1), np.expand_dims(yy, -1)], axis=-1).reshape(-1, 2)

        # points 의 dim 맞추기
        # points[:, ]

        dim_val = self.v1[eq_dim]
        dim_pts = np.full(shape=points[:, 1].shape,
                          fill_value=dim_val)

        if eq_dim == 0:
            point_list = np.stack([dim_pts, points[:, 0], points[:, 1]], axis=-1)
        elif eq_dim == 1:
            point_list = np.stack([points[:, 0], dim_pts, points[:, 1]], axis=-1)
        elif eq_dim == 2:
            point_list = np.stack([points[:, 0], points[:, 1], dim_pts], axis=-1)

        # xx, yy = np.meshgrid(xs, ys)
        return point_list


class Cube:
    def __init__(self, face_dict, num_edge):
        # each face has 4^division_level pixels (triangles)
        self.face_dict = face_dict
        self.face_list = []
        for key in self.face_dict:
            [v1, v2, v3, v4] = self.face_dict[key]
            face = Face(v1, v2, v3, v4)
            self.face_list.append(face.get_points(num_edge))


def visualize_points(points):
    # vis cube points
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    scale = 2
    # axis scale setting
    ax.set_xlim3d(-1 * scale, 1 * scale)
    ax.set_ylim3d(-1 * scale, 1 * scale)
    ax.set_zlim3d(-0.8 * scale,  0.8 * scale)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='*')
    plt.show()
    return


def normalize(vec):
    norm = np.sqrt(np.sum(vec ** 2))
    vec = vec / norm
    return vec


def inflate_cube(num_edge=20):
    face_dict = generate_face_group()
    cube = Cube(face_dict, num_edge)
    face_list = []
    for face in cube.face_list:
        face_list.append(face)

    patch_list = []
    for f in face_list:
        patch = []
        for p in f:
            p = normalize(p)
            patch.append(p)
        patch = np.array(patch)
        patch_list.append(patch)
    return patch_list


def visualize_face(point_list):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    scale = 2
    # axis scale setting
    ax.set_xlim3d(-1 * scale, 1 * scale)
    ax.set_ylim3d(-1 * scale, 1 * scale)
    ax.set_zlim3d(-0.8 * scale,  0.8 * scale)

    for patch in point_list:
        ax.scatter(patch[:, 0], patch[:, 1], patch[:, 2], marker='*')
    plt.show()
    return


if __name__ == '__main__':
    # make cube
    face_dict = generate_face_group()
    cube = Cube(face_dict, num_edge=10)
    points = np.array(cube.face_list).reshape(-1, 3)
    # vis points
    visualize_points(points)

    patch_list = inflate_cube(num_edge=58)  # this
    visualize_face(patch_list)

