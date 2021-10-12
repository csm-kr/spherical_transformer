import numpy as np
from numpy import cos, sin, sqrt
import matplotlib.pyplot as plt
from skspatial.objects import Line, Sphere

pi = np.pi
Rz = np.array(
    [
        [cos(pi * 72 / 180), - sin(pi * 72 / 180), 0],
        [sin(pi * 72 / 180), cos(pi * 72 / 180), 0],
        [0, 0, 1]
    ]
)


def transpose_1d_vec(vec):
    vec = np.expand_dims(vec, 0)
    return vec.T


def norm_vertices(vertices):
    norm = np.linalg.norm(vertices[0])
    for i, vertex in enumerate(vertices):
        vertices[i] = vertex / norm

    return vertices


def generate_icosahedron():
    # from https://en.wikipedia.org/wiki/Regular_icosahedron
    golden_ratio = (5.0 ** 0.5 + 1.0) / 2.0
    V0 = np.array([-1, golden_ratio, 0])
    V1 = np.array([1, golden_ratio, 0])
    V2 = np.array([-1, -1 * golden_ratio, 0])
    V3 = np.array([1, -1 * golden_ratio, 0])
    V4 = np.array([0, -1, golden_ratio])
    V5 = np.array([0, 1, golden_ratio])
    V6 = np.array([0, -1, -1 * golden_ratio])
    V7 = np.array([0, 1, -1 * golden_ratio])
    V8 = np.array([golden_ratio, 0, -1])
    V9 = np.array([golden_ratio, 0, 1])
    V10 = np.array([-1 * golden_ratio, 0, -1])
    V11 = np.array([-1 * golden_ratio, 0, 1])

    V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11 = \
        norm_vertices([V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11])

    return [V0, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11]


def generate_face_group():
    # from 12 vertices of icosahedron
    # this generate 20 faces of icosahedron and then return the faces as a form of dictionary
    [v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11] = \
        generate_icosahedron()

    face_dict = dict()

    # from https://en.wikipedia.org/wiki/Regular_icosahedron
    face_dict[1] = [v0, v11, v5]
    face_dict[2] = [v0, v5, v1]
    face_dict[3] = [v0, v1, v7]
    face_dict[4] = [v0, v7, v10]
    face_dict[5] = [v0, v10, v11]
    face_dict[6] = [v1, v5, v9]
    face_dict[7] = [v5, v11, v4]
    face_dict[8] = [v11, v10, v2]
    face_dict[9] = [v10, v7, v6]
    face_dict[10] = [v7, v1, v8]
    face_dict[11] = [v3, v9, v4]
    face_dict[12] = [v3, v4, v2]
    face_dict[13] = [v3, v2, v6]
    face_dict[14] = [v3, v6, v8]
    face_dict[15] = [v3, v8, v9]
    face_dict[16] = [v5, v4, v9]
    face_dict[17] = [v2, v4, v11]
    face_dict[18] = [v6, v2, v10]
    face_dict[19] = [v8, v6, v7]
    face_dict[20] = [v9, v8, v1]
    return face_dict


class Face:
    # this is a class for representing a face of icosahedron
    def __init__(self, v1, v2, v3):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3

    def get_subdivision(self):
        # get subdivision by connecting center point of each line of a triangle
        center1 = self.get_geometric_center(self.v1, self.v2)
        center2 = self.get_geometric_center(self.v2, self.v3)
        center3 = self.get_geometric_center(self.v1, self.v3)
        face1 = Face(self.v1, center1, center3)
        face2 = Face(center1, self.v2, center2)
        face3 = Face(center1, center2, center3)
        face4 = Face(center2, center3, self.v3)
        return [face1, face2, face3, face4]

    @staticmethod
    def get_geometric_center(p1, p2):
        return (p1+p2)/2

    def get_point(self):
        return [self.v1, self.v2, self.v3]

    def get_center_point(self):
        return (self.v1+self.v2+self.v3)/3


class Subdivision:
    # this is another class for representing face of an icosahedron
    # this is initialized by upper face (not divided)
    def __init__(self, upper_face: Face):
        self.face = upper_face
        self.subdivision_by_level = dict()
        self.sub_level = 0
        self.subdivision_by_level[self.sub_level] = {'0': self.face} # 0th subdivision is face itself

    def proceed_subdivision(self):
        cur_subdivision = self.subdivision_by_level[self.sub_level]
        next_subdivision = dict()
        for key in cur_subdivision.keys():
            face = cur_subdivision[key]
            sub_face_list = face.get_subdivision()
            for idx, sub_face in enumerate(sub_face_list): # divide to its next level
                new_key = key+'_'+str(idx)
                next_subdivision[new_key] = sub_face
        self.sub_level += 1
        self.subdivision_by_level[self.sub_level] = next_subdivision

    def get_highest_subdivision(self):
        return self.subdivision_by_level[self.sub_level]


class Icosahedron:
    def __init__(self, face_dict, division_level=4):
        # each face has 4^division_level pixels (triangles)
        self.face_dict = face_dict
        self.subdivision_list = []
        for key in self.face_dict:
            [v1, v2, v3] = self.face_dict[key]
            face = Face(v1, v2, v3)
            self.subdivision_list.append(Subdivision(face))

        for i in range(division_level):
            self.divide_icosahedron()

    def divide_icosahedron(self):
        for subdivision in self.subdivision_list:
            subdivision.proceed_subdivision()


def inflate_icosahedron(division_level=3):
    '''

    :param division_level:
    :return:
    '''
    # this makes icosahedron and returns a list of patch
    # each patch has shape of [4^division_level, 3]: points on sphere in 3D space
    # whole image points consists 20 * 4 ^ division_level

    face_dict = generate_face_group()
    icosahedron = Icosahedron(face_dict, division_level)
    # extrude spherePHD

    def normalize(vec):
        norm = np.sqrt(np.sum(vec ** 2))
        vec = vec / norm
        return vec

    sphere = Sphere([0, 0, 0], 1)
    eps = 1e-7
    patch_list = []
    # vertex_point_list = []
    for sub_idx, subdivision in enumerate(icosahedron.subdivision_list):
        patch = []
        sub_faces = subdivision.get_highest_subdivision()
        center_point_list = []
        for key in sub_faces.keys():
            face = sub_faces[key]
            center_point_list.append(face.get_center_point())

        for cp in center_point_list:
            line = Line([0, 0, 0], [cp[0], cp[1], cp[2]])
            point_a, point_b = sphere.intersect_line(line)
            cp = normalize(cp)
            if 1 - eps <= np.dot(point_a, cp) or np.dot(point_a, cp) <= 1 + eps:
                patch.append(point_a)
            elif 1 - eps <= np.dot(point_b, cp) or np.dot(point_b, cp) <= 1 + eps:
                patch.append(point_b)
            else:
                raise RuntimeError('point not found')
        patch = np.array(patch)
        patch_list.append(patch)

    return patch_list  # [[4 ** div_lev, 3], ...] length of list is 20


def geodesic_icosahedron(division_level=3):
    # this makes geodesic icosahedron and returns a coord list
    # whole image points consists T * 10 + 2 division_level T : (2^0, 2^2, 2^4, 2^8, 2^16)

    face_dict = generate_face_group()
    icosahedron = Icosahedron(face_dict, division_level)

    def normalize(vec):
        norm = np.sqrt(np.sum(vec ** 2))
        vec = vec / norm
        return vec

    sphere = Sphere([0, 0, 0], 1)
    eps = 1e-6
    result_vertex_point_list = []
    vertex_point_list = []
    for sub_idx, subdivision in enumerate(icosahedron.subdivision_list):

        sub_faces = subdivision.get_highest_subdivision()
        for key in sub_faces.keys():
            face = sub_faces[key]
            v1, v2, v3 = face.get_point()
            vertex_point_list.append(v1)
            vertex_point_list.append(v2)
            vertex_point_list.append(v3)

        vertex_point_list_u = np.unique(vertex_point_list, axis=0)
        for vp in vertex_point_list_u:
            line = Line([0, 0, 0], [vp[0], vp[1], vp[2]])
            point_a, point_b = sphere.intersect_line(line)
            vp = normalize(vp)
            if 1 - eps <= np.dot(point_a, vp) or np.dot(point_a, vp) <= 1 + eps:
                result_vertex_point_list.append(point_a)
            elif 1 - eps <= np.dot(point_b, vp) or np.dot(point_b, vp) <= 1 + eps:
                result_vertex_point_list.append(point_b)
            else:
                raise RuntimeError('point not found')
        # 4.683480739593506
    result_vertex_point_list = np.unique(result_vertex_point_list, axis=0)
    return result_vertex_point_list


def visualize(vertices):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for vertex in vertices:
        ax.scatter(vertex[0], vertex[1], vertex[2], marker='*')
    return


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
    patch_list = inflate_icosahedron(division_level=3)  # this
    visualize_face(patch_list)
