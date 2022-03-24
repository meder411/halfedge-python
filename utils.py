import heapq
from typing import Callable

import numpy as np
import plyfile


def pairs(lst):
    i = iter(lst)
    first = prev = item = next(i)
    for item in i:
        yield prev, item
        prev = item
    yield item, first


def triplets(lst):
    i = iter(lst)
    first = prev_prev = next(i)
    second = prev = item = next(i)
    for item in i:
        yield prev_prev, prev, item
        prev_prev = prev
        prev = item
    yield item, first, second


def compute_barycentric_coordinates(v0, v1, v2, pt):
    # Compute barycentric coordinates
    # alpha --> v0
    # beta --> v1
    # gamma --> v2

    # Compute the necessary vectors
    v1_v0 = v1 - v0
    v2_v0 = v2 - v0
    pt_v0 = pt - v0

    # Area of the parallelogram
    N = np.cross(v1_v0, v2_v0)

    # Compute the ratios of subtriangles
    beta = np.dot(np.cross(pt_v0, v2_v0), N) / np.dot(N, N)  # For v1
    gamma = np.dot(np.cross(v1_v0, pt_v0), N) / np.dot(N, N)  # For v2
    alpha = 1 - beta - gamma  # For v0
    return alpha, beta, gamma


def check_degenerate_face(vertex_indices):
    # Check for degenerate face
    if (len(set(vertex_indices)) < 3) or (len(set(vertex_indices)) !=
                                          len(vertex_indices)):
        raise ValueError("Degenerate face! Fewer than 3 unique vertices")


def compute_planar_quadric(normal, d):
    quadric = np.zeros((4, 4))
    quadric[:3, :3] = np.outer(normal, normal)
    quadric[:3, -1] = quadric[-1, :3] = normal * d
    quadric[-1, -1] = d * d
    return quadric


def points_are_collinear(p0, p1, p2):
    """
    Check if the triangle inequality is violated
    """
    len01 = np.linalg.norm(p1 - p0)
    len02 = np.linalg.norm(p2 - p0)
    len12 = np.linalg.norm(p2 - p1)

    if (len01 > len02) and (len01 > len12):
        return len01 == len02 + len12
    elif (len02 > len01) and (len02 > len12):
        return len02 == len01 + len12
    else:
        return len12 == len01 + len02


def project_point_onto_line(v0, v1, pt):
    v1_v0 = v1 - v0
    pt_v0 = pt - v0
    return v0 + v1_v0 * pt_v0.dot(v1_v0) / v1_v0.dot(v1_v0)


class Plane:

    def __init__(self, params):
        self.params = params

    def __repr__(self):
        return f"<Plane [params={self.params}]>"

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_pt_and_normal(cls, normal, pt):
        # Compute the projective offset parameter
        d = -(normal * pt).sum()
        return cls(np.append(normal, d))

    def point_to_plane_dist(self, pt):
        return (self.params[:3] * pt).sum() + self.params[-1]

    def project_point(self, pt):
        dist = self.point_to_plane_dist(pt)
        return pt + dist * normal

    def point_is_above(self, pt):
        return self.point_to_plane_dist(pt) > 0


class Quadric:
    """
    https://www.cs.cmu.edu/~./garland/Papers/quadric2.pdf
    """

    def __init__(self, plane, weight=1.0):
        self.A = weight * np.outer(plane.params[:3], plane.params[:3])
        self.b = weight * plane.params[:3] * plane.params[-1]
        self.c = weight * plane.params[-1]**2

    def __iadd__(self, other):
        self.A += other.A
        self.b += other.b
        self.c += other.c
        return self

    def __add__(self, other):
        q = Quadric(Plane(np.zeros(4)))
        q.A = self.A + other.A
        q.b = self.b + other.b
        q.c = self.c + other.c
        return q

    def apply(self, xyz):
        # Sec 3.1 in https://www.cs.cmu.edu/~./garland/Papers/quadric2.pdf
        return xyz @ self.A @ xyz + 2 * self.b @ xyz + self.c

    def is_invertible(self):
        return abs(np.linalg.det(self.A)) > 1e-6

    def optimal_point(self):
        # Sec 3.3 in https://www.cs.cmu.edu/~./garland/Papers/quadric2.pdf
        if self.is_invertible():
            opt_xyz = -np.linalg.lstsq(self.A, self.b, rcond=None)[0]
            cost = self.apply(opt_xyz)
            return opt_xyz, cost
        return None, None


def write_ply(
    output_path,
    pts,  # 3 x N
    normals=None,  # 3 x N
    rgb=None,  # 3 x N
    faces=None,  # 3 x M
    face_rgb=None,  # 3 x M
    text=False):
    names = 'x,y,z'
    formats = 'f4,f4,f4'
    if normals is not None:
        pts = np.vstack((pts, normals))
        names += ',nx,ny,nz'
        formats += ',f4,f4,f4'
    if rgb is not None:
        pts = np.vstack((pts, rgb))
        names += ',red,green,blue'
        formats += ',u1,u1,u1'
    pts = np.core.records.fromarrays(pts, names=names, formats=formats)
    el = [plyfile.PlyElement.describe(pts, 'vertex')]
    if faces is not None:
        faces = faces.astype(np.int32).T  # Next step needs M x 3 memory layout
        faces = faces.copy().ravel().view([("vertex_indices", "u4", 3)])
        el.append(plyfile.PlyElement.describe(faces, 'face'))
    if face_rgb is not None:
        el.append(plyfile.PlyElement.describe(face_rgb, 'face'))

    plyfile.PlyData(el, text=text).write(output_path)


def read_ply(path):
    plydata = plyfile.PlyData.read(path)
    vx = plydata["vertex"]["x"]
    vy = plydata["vertex"]["y"]
    vz = plydata["vertex"]["z"]
    vertices = np.dstack((vx, vy, vz))[0]
    faces = np.stack(plydata["face"]["vertex_indices"])
    return vertices, faces


class MinHeap:

    def __init__(self, key: Callable, data=()):
        self.key = key
        self.heap = [(self.key(d), d) for d in data]
        heapq.heapify(self.heap)

    def push(self, item):
        decorated = self.key(item), item
        heapq.heappush(self.heap, decorated)

    def pop(self):
        decorated = heapq.heappop(self.heap)
        return decorated[1]

    def pushpop(self, item):
        decorated = self.key(item), item
        dd = heapq.heappushpop(self.heap, decorated)
        return dd[1]

    def replace(self, item):
        decorated = self.key(item), item
        dd = heapq.heapreplace(self.heap, decorated)
        return dd[1]

    def __len__(self):
        return len(self.heap)


if __name__ == '__main__':
    min_heap = MinHeap(key=lambda x: x[0])

    min_heap.push((26, "z"))
    min_heap.push((3, "c"))
    min_heap.push((2, "b"))
    min_heap.push((3, "cc"))
    min_heap.push((4, "d"))
    import ipdb
    ipdb.set_trace()