from typing import List

import numpy as np

from utils import compute_barycentric_coordinates, project_point_onto_line, Plane, points_are_collinear


class Data:

    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __repr__(self):
        state = [
            f"{attribute}={value}"
            for attribute, value in self.__dict__.items()
        ]
        return "[" + ", ".join(state) + "]"

    def __str__(self):
        return self.__repr__()


class VertexData(Data):

    def __init__(self, XYZ: np.ndarray, **kwds):
        super().__init__(**kwds)
        self.__dict__["XYZ"] = XYZ

    def __repr__(self):
        return "<Vertex: " + super().__repr__() + ">"


class FaceData(Data):

    def __init__(self, **kwds):
        super().__init__(**kwds)

    def __repr__(self):
        return "<Face: " + super().__repr__() + ">"


# ========================================================
# ========================================================
class HalfEdge:

    def __init__(self, src, face=None, next_edge=None, twin=None):
        self.src = src
        self.face = face
        self.next_edge = next_edge
        self.twin = twin
        self.he = None
        self.id = None

    def __repr__(self):
        return f"<HalfEdge " \
            f"[id={self.id}, " \
            f"src={None if self.src is None else self.src.id}, " \
            f"face={None if self.face is None else self.face.id}, " \
            f"next={None if self.next_edge is None else self.next_edge.id}, " \
            f"twin={None if self.twin is None else self.twin.id}, " \
            f"boundary={self.is_boundary()}]" \
            ">"

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)

    def is_boundary(self):
        return self.face is None

    def is_manifold(self):
        # Any edges with 0 faces are non-manifold
        # Also edges with >2 faces, but this data structure doesn't support that, so no need to check
        if self.is_boundary() and self.twin.is_boundary():
            return False
        return True

    def midpoint(self):
        return (self.src.XYZ + self.next_edge.src.XYZ) / 2

    def length(self):
        return np.linalg.norm(self.src.XYZ - self.next_edge.src.XYZ)

    def angle(self):
        """
        Returns angle between this half edge and its next (radians)
        """

        # If points are collinear return pi
        if points_are_collinear(self.src.XYZ, self.next_edge.src.XYZ,
                                self.next_edge.next_edge.src.XYZ):
            return np.pi

        # Compute angle generally
        src = self.src.XYZ
        tgt = self.next_edge.src.XYZ
        this_line = src - tgt
        this_line /= np.linalg.norm(this_line) + 1e-12
        next_line = self.next_edge.next_edge.src.XYZ - self.next_edge.src.XYZ
        next_line /= np.linalg.norm(next_line) + 1e-12
        angle = np.arccos((this_line * next_line).sum())

        # Handle ambiguity around pi
        # If the edge normals converge, we are <pi. If not, subtract from 2pi
        this_plane = Plane.from_pt_and_normal(self.normal(), src)
        next_plane = Plane.from_pt_and_normal(self.next_edge.normal(),
                                              self.next_edge.src.XYZ)

        # Check the center point of the triangle formed by these two half edges. If it's on the positive side of both edges, this is a convex angle. If it's on the negative side of both, this is a reflex angle. If the signs are different, something is broken....
        center = (src + self.next_edge.src.XYZ +
                  self.next_edge.next_edge.src.XYZ) / 3
        if this_plane.point_is_above(center) and next_plane.point_is_above(
                center):
            return angle
        return 2 * np.pi - angle

    def normal(self):
        """
        Returns vector normal to the edge, defined as always facing inward toward its face
        """
        # If this is not a boundary edge, compute the normal by projecting the face center onto the edge. If this is a boundary edge, do the same computation on the twin and return the negative
        if self.face is not None:
            center = self.face.center()
        else:
            center = self.twin.face.center()

        normal = center - project_point_onto_line(
            self.src.XYZ, self.next_edge.src.XYZ, center)
        normal /= np.linalg.norm(normal) + 1e-12

        if self.is_boundary():
            normal *= -1

        return normal


class HEVertex(VertexData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = None
        self.halfedge = None

    def __repr__(self):
        return f"<HEVertex " \
            f"[id={self.id}, " \
            f"halfedge={None if self.halfedge is None else self.halfedge.id}]" \
            ">"

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __hash__(self):
        return hash(self.id)

    def is_boundary(self):
        # Boundary vertices have an incoming and outgoing boundary halfedge
        return self.outgoing_boundary_halfedge() is not None

    def is_connected(self):
        # A vertex is connected if it has a half edge connecting it to another vertex
        return self.halfedge is not None

    def is_manifold(self):
        # Make sure that all adjacent faces form open or closed fans. This means that the vertex must have either zero or two adjacent boundary halfedges (incoming and outgoing)
        num_boundary_edges = 0
        for out_he in self.outgoing_halfedges():
            if out_he is None:
                import ipdb
                ipdb.set_trace()
            if out_he.is_boundary():
                num_boundary_edges += 1
        for in_he in self.incoming_halfedges():
            if in_he.is_boundary():
                num_boundary_edges += 1

        # Check the number of adjacent boundary edge
        if num_boundary_edges not in [0, 2]:
            return False
        return True

    def degree(self):
        return len([vx for vx in self.adjacent_vertices()])

    def incoming_halfedges(self):
        # Get the representative outgoing half edge of the vertex
        cur_out_he = self.halfedge

        # Python do-while loop
        while True:

            # Yield current incoming halfedge
            yield cur_out_he.twin

            # Get to next outgoing halfedge by going "in" (via the twin) and "next"
            next_out_he = cur_out_he.twin.next_edge

            # Break if we've returned to the original outgoing halfedge
            if next_out_he == self.halfedge:
                break

            # Update the current outgoing halfedge
            cur_out_he = next_out_he

    def outgoing_halfedges(self):

        # Get a outgoing half edge of the vertex
        cur_out_he = self.halfedge

        # Python do-while loop
        while True:
            # Get current outgoing halfedge
            yield cur_out_he

            # Get to next outgoing halfedge by going "in" (via the twin) and "next"
            next_out_he = cur_out_he.twin.next_edge

            # Break if we've returned to the original outgoing halfedge
            if next_out_he == self.halfedge:
                break

            # Update the current outgoing halfedge index
            cur_out_he = next_out_he

    def outgoing_boundary_halfedge(self):

        # Get a outgoing half edge of the vertex
        cur_out_he = self.halfedge

        # Python do-while loop
        while True:
            if cur_out_he.is_boundary():
                return cur_out_he

            # Get to next outgoing halfedge by going "in" (via the twin) and "next"
            next_out_he = cur_out_he.twin.next_edge

            # Break if we've returned to the original outgoing halfedge
            if next_out_he == self.halfedge:
                break

            # Update the current outgoing halfedge index
            cur_out_he = next_out_he

        # If no boundary is returned, return None
        return None

    def incoming_boundary_halfedge(self):
        # Get the representative outgoing half edge of the vertex
        cur_out_he = self.halfedge

        # Python do-while loop
        while True:
            if cur_out_he.twin.is_boundary():
                return cur_out_he.twin

            # Get to next outgoing halfedge by going "in" (via the twin) and "next"
            next_out_he = cur_out_he.twin.next_edge

            # Break if we've returned to the original outgoing halfedge
            if next_out_he == self.halfedge:
                break

            # Update the current outgoing halfedge
            cur_out_he = next_out_he

        # If none are found, return None
        return None

    def adjacent_faces(self):
        # Get a outgoing half edge of the vertex
        cur_out_he = self.halfedge

        # Python do-while loop
        while True:
            # Get face of current outgoing halfedge
            if not cur_out_he.is_boundary():
                yield cur_out_he.face

            # Get to next outgoing halfedge by going "in" (via the twin) and "next"
            next_out_he = cur_out_he.twin.next_edge

            # Break if we've returned to the original outgoing halfedge
            if next_out_he == self.halfedge:
                break

            # Update the current outgoing halfedge index
            cur_out_he = next_out_he

    def adjacent_vertices(self):

        # Get a outgoing half edge of the vertex
        cur_out_he = self.halfedge

        # Python do-while loop
        while True:
            # Yield target of outgoing halfedge
            yield cur_out_he.next_edge.src

            # Get to next outgoing halfedge by going "in" (via the twin) and "next"
            next_out_he = cur_out_he.twin.next_edge

            # Break if we've returned to the original outgoing halfedge
            if next_out_he == self.halfedge:
                break

            # Update the current outgoing halfedge index
            cur_out_he = next_out_he

    def compute_normal(self):
        # Initialize normal to zeros
        self.normal = np.zeros(3, dtype=np.float64)

        # Go through neighboring faces
        for f in self.adjacent_faces():
            # Compute face normals if not yet done
            if not hasattr(f, "normal"):
                f.compute_normal()

            # Accumulate face normals
            self.normal += f.normal

        # Normalize
        self.normal /= np.linalg.norm(self.normal) + 1e-12

        # Also set the plane
        self.plane = Plane.from_pt_and_normal(self.normal.self.XYZ)

    def compute_quadric(self):
        # Accumulate fundamental error quadrics for each neighboring face
        self.quadric = Quadric(Plane(np.zeros(4)))
        for f in self.adjacent_faces():
            if not hasattr(f, "quadric"):
                f.compute_quadric()
            self.quadric += f.quadric

    def compute_neighborhood_center(self):
        # Accumulate neighboring vertex position
        xyz = np.zeros(3)
        num = 0
        for vx in self.adjacent_vertices():
            xyz += vx.XYZ
            num += 1

        # Return the average
        return xyz / num


class HEFace(FaceData):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = None
        self.halfedge = None

    def __repr__(self):
        return "<HEFace " \
            f"[id={self.id}, " \
            f"halfedge={None if self.halfedge is None else self.halfedge.id}]" \
            ">"

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return self.id != other.id

    def __contains__(self, item):
        # If checking whether a vertex is part of this face, e.g
        #   if v in face:
        if isinstance(item, HEVertex):
            return item in self.vertex_list()

        # If checking whether a halfedge is part of this face, e.g
        elif isinstance(item, HalfEdge):
            for he in self.adjacent_halfedges():
                if item == he:
                    return True
            return False

        # Only defined for vertices and halfedges
        else:
            raise TypeError(
                f"HEFace 'in' keyword only defined for checking HEVertex HalfEdge types ({type(item)})"
            )

    def __hash__(self):
        return hash(self.id)

    def num_vertices(self):
        return len(self.vertex_list())

    def face_degree(self):
        return len([face for face in self.adjacent_faces()])

    def vertex_degree(self):
        return len([vertex for vertex in self.adjacent_vertices()])

    def is_degenerate(self):
        # If there are less than 3 unique IDs or if there is a duplicate vertex ID, mark face as degenerate
        return (len(set(self.vertex_list())) < 3) or (len(
            set(self.vertex_list())) != len(self.vertex_list()))

    def is_boundary(self):
        for he in traverse_edges(self.halfedge):
            if he.is_boundary():
                return True
        return False

    def barycentric_coordinates(self, XYZ):
        # Compute barycentric coordinates
        # alpha --> A
        # beta --> B
        # gamma --> C

        # Only valid for triangular faces
        if self.num_vertices() != 3:
            raise TypeError(
                "Can only compute barycentric coordinates on triangular faces")

        vertex_list = self.vertex_list()
        return compute_barycentric_coordinates(vertex_list[0].XYZ,
                                               vertex_list[1].XYZ,
                                               vertex_list[2].XYZ, XYZ)

    def contains_point(self, XYZ):
        # Check if XYZ coordinates fall in this face. Does a point-in-triangle test using barycentirc coordinates
        alpha, beta, gamma = self.barycentric_coordinates(XYZ)
        return (0.0 <= alpha <= 1.0) and (0.0 <= beta <= 1.0) and (0.0 <= gamma
                                                                   <= 1.0)

    def center(self):
        # If this face is degenerate, just return the src of the halfedge
        if self.is_degenerate():
            return self.halfedge.src.XYZ

        # Otherwise average the vertices
        center = np.zeros(3)
        num_adjacent_vertices = 0
        for vx in self.adjacent_vertices():
            center += vx.XYZ
            num_adjacent_vertices += 1

        return center / num_adjacent_vertices

    def vertex_list(self):
        vx_list = []
        for vx in self.adjacent_vertices():
            vx_list.append(vx)
        return vx_list

    def boundary_halfedge(self):
        for he in self.adjacent_halfedges():
            if he.twin.is_boundary():
                return he.twin

        return None

    def adjacent_halfedges(self):
        # Circulate the edges
        for he in traverse_edges(self.halfedge):

            # Yield each one
            yield he

    def adjacent_faces(self):
        # Circulate the edges
        for he in traverse_edges(self.halfedge):

            # If the twin of the edge is not a boundary it has a face
            if not he.twin.is_boundary():

                # Yield face index
                yield he.twin.face

    def adjacent_vertices(self):

        # Circulate the edges
        for he in traverse_edges(self.halfedge):

            # Yield target vertex
            yield he.next_edge.src

    def compute_normal(self):
        # Get 3 sequential vertices
        vxs = []
        itt = self.adjacent_vertices()
        for i in range(3):
            vxs.append(next(itt))

        # Compute the normal as the cross product of lines between vertices
        line01 = vxs[0].XYZ - vxs[1].XYZ
        line21 = vxs[2].XYZ - vxs[1].XYZ
        self.normal = np.cross(line21, line01)

        # Normalize the normal
        self.normal /= (np.linalg.norm(self.normal) + 1e-12)

        # Create a plane as well
        self.plane = Plane.from_pt_and_normal(self.normal, vxs[0].XYZ)

    def compute_quadric(self):
        # Compute face normal is not yet done
        if not hasattr(self, "normal"):
            self.compute_normal()

        # Compute the quadric
        self.quadric = Quadric(self.plane)


class BoundaryLoop(Data):
    """
    Class to store data about boundary loops
    """

    def __init__(self, halfedge):
        if not halfedge.is_boundary():
            raise ValueError(
                "Cannot create a boundary loop with an interior halfedge!")
        self.halfedge = halfedge

    def __len__(self):
        # If no halfedge, the boundary loop is len 0
        if self.halfedge is None:
            return 0

        # Otherwise count the edges
        length = 0
        for he in traverse_edges(self.halfedge):
            length += 1
        return length

    def __repr__(self):
        return f"<BoundaryLoop " \
            f"[representative_edge={self.halfedge}, " \
            f"len={len(self)}]" \
            ">"


def traverse_edges(he):

    # Do while loops, Python style
    start = he
    while True:
        # Return the half edge
        yield he

        # If the next half edge is back to the start, break the loop
        if he.next_edge == start:
            break

        # Otherwise, advance to the next half edge
        he = he.next_edge