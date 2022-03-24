import sys
from dataclasses import dataclass

import numpy as np
from scipy import spatial

from utils import MinHeap, Quadric, Plane


def quadric_error_function(src, tgt, halfedge):

    # If this is a boundary edge, form the boundary condition quadric
    if halfedge is not None:
        if halfedge.is_boundary() or halfedge.twin.is_boundary():

            # Get the non-boundary edge that connects them and use it's face's normal
            if halfedge.is_boundary():
                face_normal = halfedge.twin.face.normal
            else:
                face_normal = halfedge.face.normal

            # Compute the normal of the plane perpendicular to the boundary face for which this pair forms an edge
            boundary_normal = np.cross(tgt.XYZ - src.XYZ, face_normal)

            # Compute the boundary quadric and weight it heavily
            boundary_quadric = Quadric(Plane.from_pt_and_normal(
                boundary_normal, src.XYZ),
                                       weight=1e3)

            # Add the boundary quadric to both vertices
            src.quadric += boundary_quadric
            tgt.quadric += boundary_quadric

    # Optimal quadric is the sum of the src and tgt quadrics
    opt_quadric = src.quadric + tgt.quadric

    # Try to compute the optimal point and its cost
    opt_xyz, opt_cost = opt_quadric.optimal_point()

    # If the quadric was not invertible, use the min cost of either vertex or their midpoint on the edge
    if opt_xyz is None:
        mid_pt = (src.XYZ + tgt.XYZ) / 2
        src_cost = opt_quadric.apply(src.XYZ)
        tgt_cost = opt_quadric.apply(tgt.XYZ)
        mid_cost = opt_quadric.apply(mid_pt)

        # Return the minimum cost point
        if (src_cost < tgt_cost) and (src_cost < mid_cost):
            return src.XYZ, src_cost
        elif (tgt_cost < src_cost) and (tgt_cost < mid_cost):
            return tgt.XYZ, tgt_cost
        else:
            return mid_pt, mid_cost

    # Return the optimal cost point
    return opt_xyz, opt_cost


class Pair:
    # Class for a vertex pair

    def __init__(self,
                 src: "HEVertex",
                 tgt: "HEVertex",
                 halfedge: "HalfEdge" = None,
                 cost_function=quadric_error_function):
        self.src = src
        self.tgt = tgt
        self.halfedge = halfedge  # None if not connected
        self.opt_xyz, self.cost = cost_function(src, tgt, halfedge)

    def __lt__(self, other):
        return self.cost < other.cost

    def is_safe_merge(self):
        # Safe to merge unconnected vertices, generally
        if self.halfedge is None:
            return True

        # For connected vertices, they must have exactly two edge-adjacent vertices in common. So we gather the two sets of adjacent vertex IDs and look at the size of their intersection
        src_adj_vx_id = set()
        for vx in self.src.adjacent_vertices():
            src_adj_vx_id.add(vx.id)
        tgt_adj_vx_id = set()
        for vx in self.tgt.adjacent_vertices():
            tgt_adj_vx_id.add(vx.id)
        if len(src_adj_vx_id.intersection(tgt_adj_vx_id)) == 2:
            return True

        # Otherwise, return false
        return False

    def merge_vertices(self):
        """
        Contracts the edge by moving all tgt edges to the src and updating the src vertex
        """

        # =======================================
        # Note elements to remove
        # =======================================
        # Identify the faces to remove (only if an edge collapse)
        if self.halfedge is not None:
            removed_faces = [self.halfedge.face, self.halfedge.twin.face]
        else:
            removed_faces = []

        # Mark the half edges interior to the faces being removed, if relevant
        removed_halfedges = []
        if removed_faces:
            for he in removed_faces[0].adjacent_halfedges():
                removed_halfedges.append(he)
            for he in removed_faces[1].adjacent_halfedges():
                removed_halfedges.append(he)

        # Just removing the tgt vertex
        removed_vertices = [tgt]

        # =======================================
        # Re-link halfedges
        # =======================================
        # Set the origin of outgoing halfedges in the tgt vertex to instead originate from the src vertex as long as they're not part of the faces being removed. Also update src's halfedge as we may have just removed it.
        for out_he in tgt.outgoing_halfedges():
            if (out_he.face.id != removed_faces[0].id) and (
                    out_he.face.id != removed_faces[1].id):
                out_he.src = src
                src.halfedge = out_he

        # Set up twin relations for one removed face (a picture really helps here)
        self.halfedge.next_edge.twin.twin = self.halfedge.next_edge.next_edge.twin
        self.halfedge.next_edge.next_edge.twin = self.halfedge.next_edge.twin.twin

        # Set up twin relations for the other removed face (a picture also really helps here)
        self.halfedge.twin.next_edge.twin.twin = self.halfedge.twin.next_edge.next_edge.twin
        self.halfedge.twin.next_edge.next_edge.twin = self.halfedge.twin.next_edge.twin.twin

        # Update wing vertices' halfedges to point to halfedges that are guaranteed to still exist after this procedure
        self.halfedge.next_edge.tgt.halfedge = self.halfedge.next_edge.twin
        self.halfedge.twin.next_edge.tgt.halfedge = self.halfedge.twin.next_edge.twin

        # =======================================
        # Update src vertex
        # =======================================
        src.XYZ = self.opt_xyz

        return removed_faces, removed_halfedges, removed_vertices


class EdgeCollapse:

    def __init__(self, threshold=0.0):

        # Min heap for priority queue
        self.min_heap = MinHeap(key=lambda x: x.cost)

        self.threshold = threshold
        self.kdtree_leaf_size = 1000

    def create_pair(self, mesh, src, src_idx, tgt, tgt_idx):

        # Optimal quadric is the sum of the src and tgt quadrics
        opt_quadric = src.quadric + tgt.quadric

        # If summed quadric is invertible, compute optimal vertex solution
        if np.linalg.cond(opt_quadric) < 1 / sys.float_info.epsilon:
            opt_v = np.linalg.inv(opt_quadric)[:, -1]
        # Otherwise, take the average of the pair
        # TODO(Marc): Better minimizer?
        else:
            opt_v = ((src.XYZ + tgt.XYZ) / 2.0).append(1.0)

        # Compute cost
        cost = opt_v.dot(opt_quadric).dot(opt_v)

        opt_v = opt_v[:3] / opt_v[-1]

        # Create a pair object. Always store edge vertices as ascending tuple.
        pair = Pair(src_idx if src_idx < tgt_idx else tgt_idx,
                    tgt_idx if src_idx < tgt_idx else src_idx, opt_v, cost)

        return pair

    def contract_edge(self, mesh, src_idx, tgt_idx, opt_XYZ):
        """
        Changes the XYZ of src_idx to be the new optimal value, attaches all halfedges of tgt_idx to those at src_idx, and removes the tgt_idx vertex
        """
        # Get the vertex elements
        src = mesh.vertex(src_idx)
        tgt = mesh.vertex(tgt_idx)

        # Change the src vertex to the new location
        src.XYZ = opt_XYZ

        # for outmesh.vertex_outgoing_halfedges(tgt_idx)

    def collapse(self, mesh):

        # ======================================================================
        # 1. Compute the error quadrics for all vertices
        # ======================================================================
        for _, v in mesh.vertices.items():
            v.compute_quadric()
            v.pairs = []  # Create list to hold pairs

        # Create the set of valid pairs defined by edge relations.
        edge_set = set()
        for _, he in mesh.halfedges.items():

            # Use a canonical vertex ordering for edges
            if he.src.id < he.tgt.id:
                src = he.src
                tgt = he.tgt
            else:
                src = he.tgt
                tgt = he.src

            # Edges are always ordered as ascending vertex indices
            edge = (src, tgt)

            # If this edge is already represented, continue
            if edge in edge_set:
                continue

            # Create a pair
            pair = Pair(src, tgt, he)

            # Add pair to priority queue
            self.min_heap.push(pair)

            # Also track the pair from within the vertices
            src.pairs.append(pair)
            tgt.pairs.append(pair)

            # Add the vertex pairs that represent this edge
            edge_set.add(edge)

        # If the threshold is non-zero, we also look at unconnected, nearby vertices to find pairs
        if (self.threshold > 0.0) and (mesh.num_boundary_vertices() > 0):
            # Gather up the boundary vertices and their respective indices (only ones potentially being contracted)
            boundary_pts = np.array([
                v.XYZ for vx_id, v in mesh.vertices.items()
                if v.is_boundary_vertex()
            ])
            ids = np.array([
                vx_id for vx_id, v in mesh.vertices.items()
                if v.is_boundary_vertex()
            ])

            # Create KDTree for fast spatial lookup
            kdtree = spatial.KDTree(boundary_pts,
                                    leafsize=self.kdtree_leaf_size)

            # Query all boundary vertices against themselves to see if they are within the threshold distance
            queries = kdtree.query_ball_tree(kdtree, r=self.threshold)

            # Go through each matched vertex (skipping first one as it's identity)
            for src_id, matches in zip(ids, queries):
                for tgt_id in matches[1:]:

                    # Use a canonical vertex ordering for edges
                    if src_id < tgt_id:
                        src = mesh.vertex(src_id)
                        tgt = mesh.vertex(tgt_id)
                    else:
                        src = mesh.vertex(tgt_id)
                        tgt = mesh.vertex(src_id)

                    # Edges are always ordered as ascending vertex indices
                    edge = (src, tgt)

                    # If this edge is already represented, continue
                    if edge in edge_set:
                        continue

                    # Create a pair
                    pair = Pair(src, tgt)

                    # Add pair to priority queue
                    self.min_heap.push(pair)

                    # Also track the pair from within the vertices
                    src.pairs.append(pair)
                    tgt.pairs.append(pair)

                    # Add the vertex pairs that represent this edge
                    edge_set.add(edge)

        import ipdb
        ipdb.set_trace()

        for i in range(3):
            min_cost_pair = self.min_heap.pop()

            if not min_cost_pair.is_safe_merge():
                continue

            removed_faces, removed_halfedges, removed_vertices = pair.merge_vertices(
            )

            for face in removed_faces:
                mesh.faces.pop(face.id)
            for halfedge in removed_halfedges:
                mesh.halfedges.pop(halfedge.id)
            for vertex in removed_vertices:
                mesh.vertices.pop(vertex.id)