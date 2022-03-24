from dataclasses import dataclass
from enum import Enum, auto
import heapq
import sys
import warnings

from scipy import spatial
import numpy as np

from he_mesh import HalfEdgeMesh
from edge_collapse import EdgeCollapse
from he_elements import BoundaryLoop, HalfEdge, HEVertex, HEFace, traverse_edges
from utils import triplets, pairs, check_degenerate_face, Quadric, Plane, write_ply, read_ply, points_are_collinear


class SubdivisionType(Enum):
    MIDPOINT = auto()
    LOOP = auto()


class HalfEdgeTriangleMesh(HalfEdgeMesh):

    def subdivide(self, subdivision_type):

        if subdivision_type not in [
                SubdivisionType.MIDPOINT, SubdivisionType.LOOP
        ]:
            raise ValueError(
                f"Invalid subdivision type! Only MIDPOINT and LOOP are valid for triangle meshes. ({subdivision_type})"
            )

        # Set to track split half edges
        split_halfedges = set()

        # Get the original half edge info
        original_he = {he for _, he in self.halfedges.items()}

        # Split all original half edges
        # When this is done, each face should have 6 vertices instead of 3
        for he in original_he:

            # Because splitting an edge affects both a half edge and its twin, keep track of what has been split so we don't inadvertently split a twin again
            if (he in split_halfedges) or (he.twin in split_halfedges):
                continue

            # Add the split edges to the set. Do this before splitting because twinship gets reconfigured.
            split_halfedges.add(he)
            split_halfedges.add(he.twin)

            # Split the edge
            self.split_edge(he.id)

        # Get the original face info
        original_faces = [(face_id, face)
                          for face_id, face in self.faces.items()]

        # Go through each face, creating the new faces and linkages by splitting the face
        for face_id, face in original_faces:

            # Get the new and old vertices (ordered)
            new_vx = []
            old_vx = []
            for he in face.adjacent_halfedges():
                if he not in original_he:
                    new_vx.append(he.src)
                else:
                    old_vx.append(he.src)

            # Interpolate vertices if doing Loop vertices
            if subdivision_type == SubdivisionType.LOOP:

                # First interpolate the odd vertices
                for nvx in new_vx:
                    a = nvx.halfedge.next_edge.src
                    b = nvx.halfedge.twin.next_edge.twin.src
                    if nvx.is_boundary():
                        nvx.XYZ = (a.XYZ + b.XYZ) / 2
                    else:
                        c = nvx.halfedge.next_edge.next_edge.next_edge.src
                        d = nvx.halfedge.twin.next_edge.next_edge.next_edge.next_edge.src
                        nvx.XYZ = (3 / 8) * (a.XYZ + b.XYZ) + (c.XYZ +
                                                               d.XYZ) / 8

                # Then interpolate the even vertices
                for ovx in old_vx:
                    if ovx.is_boundary():
                        a = ovx.halfedge.next_edge.src
                        b = ovx.halfedge.twin.next_edge.twin.src
                        ovx.XYZ = (a.XYZ + b.XYZ) / 8 + 3 * ovx.XYZ / 4
                    else:
                        beta = 3 / (8 *
                                    ovx.degree()) if ovx.degree() > 3 else (3 /
                                                                            16)
                        accum = np.zeros(3)
                        for adjvx in ovx.adjacent_vertices():
                            accum += adjvx.XYZ
                        ovx.XYZ = ovx.XYZ * (
                            1 - beta * ovx.degree()) + beta * accum

            # Split the face 3 times between the new vertices
            face0, face1 = self.split_face(face.id, new_vx[0].id, new_vx[1].id)
            face = face0 if face0.vertex_degree() > 3 else face1
            face0, face1 = self.split_face(face.id, new_vx[1].id, new_vx[2].id)
            face = face0 if face0.vertex_degree() > 3 else face1
            face0, face1 = self.split_face(face.id, new_vx[2].id, new_vx[0].id)


    def delaunay_triangulation(self):
        """
        DeWall algorithm?
        """
        pass

    def quadric_edge_collapse(self,
                              tgt_num_faces,
                              thresh=0.1,
                              kdtree_leaf_size=10):

        # Data class for a vertex pair
        @dataclass
        class Pair:
            src: int
            tgt: int
            cost: float

        # Min heap for priority queue
        min_heap = MinHeap(key=lambda x: x.cost)

        # ======================================================================
        # 1. Compute the error quadrics for all vertices
        # ======================================================================
        for vertex_idx, v in enumerate(self.vertices):
            v.quadric = self.compute_vertex_quadric(vertex_idx)
            v.pairs = []  # Create list to hold pairs

        # ======================================================================
        # 2. Select all valid pairs and compute optimal contraction target
        # ======================================================================
        # Create the set of valid pairs defined by edge relations.
        pair_set = set()
        for he in self.halfedges:

            # Vertices representing this edge (and its twin)
            src = self.vertex(he.src)
            tgt = self.vertex(he.tgt)

            # Optimal quadric is the sum of the src and tgt quadrics
            opt_quadric = src.quadric + tgt.quadric

            # If summed quadric is invertible, compute optimal vertex solution
            if np.linalg.cond(opt_quadric) < 1 / sys.float_info.epsilon:
                opt_v = np.linalg.inv(opt_quadric)[:, -1]
            # Otherwise, take the average of the pair
            # TODO(Marc): Better minimizer?
            else:
                opt_v = (src.XYZ + tgt.XYZ) / 2.0

            # Compute cost
            cost = opt_v.dot(opt_quadric).dot(opt_v)

            # Create a pair object. Always store edge vertices as ascending tuple.
            pair = Pair(min(he.src, he.tgt), max(he.src, he.tgt), cost)

            # Add pair to priority queue
            min_heap.push(pair)

            # Also track the pair from within the vertices
            src.pairs.append(pair)
            tgt.pairs.append(pair)

            # If this edge is not yet represented by its twin, add its vertices to the valid pair list
            if (tgt, src) not in edge_set:

                # Add the vertex pairs that represent this edge
                edge_set.add((src, tgt))

        # If the threshold is non-zero, we also look at unconnected, nearby vertices to find pairs
        if thresh > 0.0:
            # Gather up the boundary vertices and their respective indices (only ones potentially being contracted)
            boundary_pts = np.stack([
                v.XYZ for i, v in enumerate(self.vertices)
                if self.is_boundary(i)
            ])
            idx = np.array([
                i for i, v in enumerate(self.vertices) if self.is_boundary(i)
            ])

            # Create KDTree for fast spatial lookup
            kdtree = spatial.KDTree(boundary_pts, leafsize=kdtree_leaf_size)

            # Query all boundary vertices against themselves to see if they are within the threshold distance
            queries = kdtree.query_ball_tree(kdtree, r=thresh)

            # Go through each matched vertex (skipping first one as its identity)
            for src, matches in zip(idx, queries):
                for tgt in matches[1:]:
                    edge_set.add((src, tgt))

        # ======================================================================
        # 3. Compute optimal contraction target for each pair and compute cost
        # ======================================================================

        # For each pair
        edge_list = list(edge_set)
        error = []
        for (src, tgt) in edge_list:

            # Sum the error quadric for each vertex
            src_quadric = self.vertex_quadrics[src]
            tgt_quadric = self.vertex_quadrics[tgt]
            opt_quadric = src_quadric + tgt_quadric

            # If summed quadric is invertible, compute optimal solution
            if np.linalg.cond(opt_quadric) < 1 / sys.float_info.epsilon:
                opt_v = np.linalg.inv(opt_quadric)[:, -1]

            # Otherwise, take the average of the pair
            # TODO(Marc): Better minimizer?
            else:
                opt_v = (self.vertex(src).XYZ + self.vertex(tgt).XYZ) / 2.0

            # Compute error
            error.append(opt_v.dot(opt_quadric).dot(opt_v))

        # ======================================================================
        # 4. Create min heap keyed on costs
        # ======================================================================
        heap = []
        for edge, error in zip(edge_list, error):
            heapq.heappush(heap, error)

        # ======================================================================
        # 5. Iteratively remove min cost pair and update relevant pairs
        # ======================================================================


if __name__ == '__main__':
    # vertices = np.array(
    #     [1.0, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0,
    #      -1]).reshape(-1, 3)
    # faces = np.array([
    #     # 0, 1, 2, 0, 2, 4, 0, 4, 5, 0, 5, 1, 3, 1, 5, 3, 5, 4, 3, 4, 2, 3, 2, 1
    #     0,
    #     2,
    #     4,
    #     0,
    #     4,
    #     5,
    #     0,
    #     5,
    #     1,
    #     3,
    #     1,
    #     5,
    #     3,
    #     5,
    #     4,
    #     3,
    #     4,
    #     2,
    #     3,
    #     2,
    #     1
    # ]).reshape(-1, 3)
    # vertices, faces = read_ply("/home/meder/Downloads/hole2.ply")
    # vertices, faces = read_ply("/home/meder/Downloads/quad.ply")
    vertices = np.array([
        [0, 0, 0],
        [0, 1.0, 0],
        [1.0, 1.0, 0],
        [1.0, 2.0, 0],
        [2.0, 2.0, 0],
        [2.0, 1.0, 0],
        [2.0, 0.0, 0],
    ])
    faces = np.array([[0, 1, 2, 3, 4, 5, 6]])
    hemesh = HalfEdgeTriangleMesh(vertices, faces)
    face = hemesh.faces[0]
    import ipdb
    ipdb.set_trace()
    hemesh.triangulate(face)
    hemesh.write_mesh("triangle_mesh.ply")
    hemesh.subdivide(SubdivisionType.LOOP)
    # for f in hemesh.faces_adjacent_to_face(2):
    #     print(f)
    # loop = hemesh.find_boundary_loops()
    # hemesh.fill_hole(loop[0])
    # hemesh.write_mesh("hemesh_output_hole.ply")
    hemesh.write_mesh("triangle_mesh_subdiv.ply")