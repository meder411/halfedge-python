import warnings

import numpy as np

from he_elements import BoundaryLoop, HalfEdge, HEVertex, HEFace, traverse_edges
from utils import pairs, check_degenerate_face, write_ply, compute_barycentric_coordinates

# ========================================================
# ========================================================


class HalfEdgeMesh:

    def __init__(self, vertices, faces):

        self.vertices = dict()
        self.edges = dict()
        self.halfedges = dict()
        self.faces = dict()

        self._next_vx_id = 0
        self._next_face_id = 0
        self._next_he_id = 0

        # Internal data structure to track existing edges during construction
        self._paired_vertices = dict()
        self._boundary_edges = dict()

        # Construct the half edge mesh
        self._create_halfedge_mesh(vertices, faces)

    # ==============================================================
    # Internal construction methods
    # ==============================================================
    def _create_halfedge_mesh(self, vertices, faces):
        """
        Parameters
        ----------
        vertices : np.ndarray
            (N, 3)
        faces : np.ndarray
            (N, M)

        """
        # Add vertices
        for v in vertices:
            self.add_vertex(v)

        # Add faces (and half edges)
        # This face creation routine is not guaranteed to be manifold through its entire sequence due to the unordered face list. At any point in time in this creation routine, even a manifold mesh may be in a non-manifold state. Manifold creation would require an incremental growing algorithm. Such an algorithm can probably be implemented efficiently via a disjoint set data structure, but it's not entirely clear what benefits that would add beyond free connected components identification and some code reduction.
        # boundary edges need to be gathered at the end.
        for f in faces:
            self._create_face(f)

        # Link boundary vertices
        # If the map of boundary vertices doesn't exist, create it
        if not self._boundary_edges:

            # Create mapping of (source vertex --> boundary half edge)
            self._boundary_edges = {
                he.src: he
                for _, he in self.halfedges.items() if he.is_boundary()
            }

        # Resolve all unlinked boundary edges. This assumes that the mesh is manifold and thus any boundary vertex has exactly 0 or 2 boundary edges and they are in sequence.
        for (src, tgt), he in self._paired_vertices.items():

            # Link the boundary half edge outgoing from the target of the current boundary half edge
            if he.is_boundary():
                he.next_edge = self._boundary_edges[tgt]

        # Check for manifoldness
        if not self.is_manifold():
            warnings.warn("Mesh is non-manifold!")

    def add_vertex(self, xyz):
        return self._new_vertex(xyz)

    def add_face(self, vertex_ids):
        # Identify all incoming boundary edges to this new face
        incoming_boundary_halfedges = dict()
        for vx_id in (vertex_ids):
            vx = self.vertex(vx_id)
            incoming_boundary_halfedges[vx] = vx.incoming_boundary_halfedge()

        # Create face
        face = self._create_face(vertex_ids)

        # If any of the twins of the new halfedges added for this face are boundaries, make sure they are linked accordingly
        for src_vx_id, tgt_vx_id in pairs(vertex_ids):
            src = self.vertex(src_vx_id)
            tgt = self.vertex(tgt_vx_id)
            twin_he = self._paired_vertices[(tgt, src)]

            if twin_he.is_boundary():
                twin_he.next_edge = self._boundary_edges[src]
                self._boundary_edges[tgt] = twin_he
                incoming_boundary_halfedges[tgt].next_edge = twin_he

        # Return the new face
        return face

    def remove_vertex(self, vx_id):
        # Pop the vertex from the list
        vx = self.vertices.pop(vx_id)

        # Get refs to all incoming and outgoing halfedges
        in_he = [he for he in vx.incoming_halfedges()]
        out_he = [he for he in vx.outgoing_halfedges()]

        # Remove all faces incident to vertex
        for face in vx.adjacent_faces():
            self.remove_face(face.id)

        # Now remove the incoming and outgoing halfedges from the mesh unless they are twins to a remaining face
        for he in (in_he + out_he):
            self.halfedges.pop(he.id)

    def remove_face(self, face_id):
        # Pop the face from the list
        face = self.faces.pop(face_id)

        # Disassociate all half edges that formerly were part of this face
        for he in face.adjacent_halfedges():
            he.face = None

    def _new_face(self):

        # Create a face and give it an id
        face = HEFace()
        face.id = self._next_face_id

        # Add it to the face dict
        self.faces[face.id] = face

        # Increment the next ID
        self._next_face_id += 1

        return face

    def _new_vertex(self, xyz):
        # Create a half edge vertex
        vertex = HEVertex(xyz)

        # Give it an ID
        vertex.id = self._next_vx_id

        # Add it to vertex dict
        self.vertices[vertex.id] = vertex

        # Increment ID
        self._next_vx_id += 1

        # Return the vertex
        return vertex

    def _new_halfedge(self, src, tgt, face=None):
        # Create the half edge
        he = HalfEdge(src, face=face)

        # Add an ID
        he.id = self._next_he_id

        # Increment the ID tracker
        self._next_he_id += 1

        # Add the half edge to the dict
        self.halfedges[he.id] = he

        # Track that it exists
        self._paired_vertices[(src, tgt)] = he

        return he

    def _create_face(self, vertex_ids):

        # Check existence of vertices
        for vx_id in vertex_ids:
            if vx_id not in self.vertices:
                raise ValueError(
                    f"Face creation error! Vertex ID not found! ({vx_id})")

        # Check if face info is degenerate
        check_degenerate_face(vertex_ids)

        face = self._new_face()

        # Container to store the half edges
        he_list = []

        # Go over all adjacent face vertex pairs (wrapping around)
        for src_vx_id, tgt_vx_id in pairs(vertex_ids):

            # Reference the vertices
            src_vx = self.vertex(src_vx_id)
            tgt_vx = self.vertex(tgt_vx_id)

            # Add the half edge to the mesh
            he = self._create_halfedge(src_vx, tgt_vx, face=face)
            he_list.append(he)

            # Set src vertex edge
            src_vx.halfedge = he

            # Add the representative halfedge to the face (will lead to the most recent halfedge being the representative one)
            face.halfedge = he

        # Link the half edges in sequence
        for cur_he, next_he in pairs(he_list):
            cur_he.next_edge = next_he

        # Return the new face
        return face

    def _create_halfedge(self, src, tgt, face):

        # ===============================================================
        # Creation case 1: Half edge already exists between (src, tgt)
        # ===============================================================
        # Check if there already exists a half edge for each vertex pair in this face. This can arise when:
        #   (a) there already exists a boundary half edge, or
        #   (b) when there is a non-manifold mesh.
        # For (a): If one exists and is a boundary edge, we reference it.
        # For (b) If one exists and it is *not* a boundary edge, we have a non-manifold edge, so throw an error (for now). TODO: Handle non-manifold edges.
        if (src, tgt) in self._paired_vertices:

            # Query the half edge index
            he = self._paired_vertices[(src, tgt)]

            # Error if it's not a boundary, because that means it's a non-manifold edge
            if not he.is_boundary():
                raise ValueError("Mesh has non-manifold edges!")

            # Update the associated face, which also marks it as an interior edge
            he.face = face

            # Remove this edge from the boundary edge map once its been initialized
            if self._boundary_edges:
                self._boundary_edges.pop(src)

        # ==============================================================
        # Creation case 2: No half edge exists yet between (src, tgt)
        # ==============================================================
        # If it doesn't exist yet, create a new half edge. We create twins automatically as well.
        else:
            # Create half edge
            he = self._new_halfedge(src, tgt, face)

            # =====================================================
            # Setting the twin (i.e. (tgt, src))
            # ======================================================

            # Create a boundary half edge as the twin
            twin = self._new_halfedge(tgt, src, None)

            # Set the twin connections
            he.twin = twin
            twin.twin = he

        # Return the new half edge
        return he

    def remove_unconnected_vertices(self):
        for vx_id, v in self.vertices.items():
            if not v.is_connected():
                self.vertices.pop(vx_id)

    def write_mesh(self, fname):

        # Map vertex ID to index (necessary if the mesh has been modified in some way)
        id_to_idx = dict()

        vertex_list = []
        for vx_id, vx in self.vertices.items():
            vertex_list.append(vx.XYZ)
            id_to_idx[vx_id] = len(vertex_list) - 1
        pts = np.array(vertex_list)

        faces = []
        for face_id, face in self.faces.items():
            vxs = face.vertex_list()
            faces.append([id_to_idx[vx.id] for vx in vxs])
        faces = np.array(faces)

        write_ply(fname, pts=pts.T, faces=faces.T, text=True)

    # ==============================================================
    # Properties
    # ==============================================================
    def vertex(self, vx_id):
        return self.vertices[vx_id]

    def face(self, face_id):
        return self.faces[face_id]

    def halfedge(self, he_id):
        return self.halfedges[he_id]

    # ==============================================================
    # Booleans
    # ==============================================================
    def is_manifold(self):
        # Any edges with 0 faces are non-manifold
        # Also edges with >2 faces, but this data structure doesn't even support that, so no need to check
        for _, he in self.halfedges.items():
            if not he.is_manifold():
                return False

        # Check for non-manifold vertices
        for _, v in self.vertices.items():
            if v.is_connected() and not v.is_manifold():
                return False

        # If both checks pass, this is a manifold mesh
        return True

    # ==============================================================
    # Counting functions
    # ==============================================================
    def num_faces(self):
        return len(self.faces)

    def num_vertices(self):
        return len(self.vertices)

    def num_halfedges(self):
        return len(self.halfedges)

    def num_edges(self):
        # All edges have two half edges
        return len(self.halfedges) // 2

    def num_boundary_vertices(self):
        return len([v for _, v in self.vertices.items() if v.is_boundary()])

    def num_interior_vertices(self):
        return len(
            [v for _, v in self.vertices.items() if not v.is_boundary()])

    def num_boundary_halfedges(self):
        return len(
            [he for _, he in self.halfedges.items() if he.is_boundary()])

    def num_interior_halfedges(self):
        return len(
            [he for _, he in self.halfedges.items() if not he.is_boundary()])

    def euler_characteristic(self):
        # |V| - |E| + |F|
        return self.num_vertices() - self.num_edges() + self.num_faces()

    # ==============================================================
    # Normal computation
    # ==============================================================
    def compute_face_normals(self):
        for _, f in self.faces.items():
            f.compute_normal()

    def compute_vertex_normals(self):
        for _, v in self.vertices:
            v.compute_normal()

    # ==============================================================
    # Circulators
    # ==============================================================
    # Iterate over half edges incoming to vertex
    def vertex_incoming_halfedges(self, vx_id):
        # Look up the vertex
        vertex = self.vertex(vx_id)

        for in_he in vertex.incoming_halfedges():
            yield in_he

    # Iterate over half edges outgoing from vertex
    def vertex_outgoing_halfedges(self, vx_id):
        # Look up the vertex
        vertex = self.vertex(vx_id)

        for out_he in vertex.outgoing_halfedges():
            yield out_he

    # Iterate over faces adjacent to vertex
    def faces_adjacent_to_vertex(self, vx_id):
        # Look up the vertex
        vertex = self.vertex(vx_id)

        for f in vertex.adjacent_faces():
            yield f

    # Iterate over vertices adjacent to vertex
    def vertices_adjacent_to_vertex(self, vx_id):
        # Look up the vertex
        vertex = self.vertex(vx_id)

        for v in vertex.adjacent_vertices():
            yield v

    # Iterate over half edges assigned to face
    def face_halfedges(self, face_id):

        # Look up the face
        face = self.faces[face_id]

        for he in face.adjacent_halfedges():
            yield he

    # Iterate over faces adjacent to face
    def faces_adjacent_to_face(self, face_id):
        # Look up the face
        face = self.faces[face_id]

        for f in face.adjacent_faces():
            yield f

    # Iterate over vertices adjacent to face
    def vertices_adjacent_to_face(self, face_id):

        # Look up the face
        face = self.faces[face_id]

        for v in face.adjacent_vertices():
            yield v

    # ===============================================================
    # Split Elements
    # ===============================================================

    def split_face(self, face_id, vx0_id, vx1_id):
        """
        Splits the face into 2 faces by connecting vx0 and vx1. Must be a face with degree >4.
        """

        # Reference elements
        face = self.face(face_id)
        vx0 = self.vertex(vx0_id)
        vx1 = self.vertex(vx1_id)

        # Check for splitting conditions
        if (vx0 not in face) or (vx1 not in face):
            raise ValueError(
                "Cannot split face! At least one of the requested vertices is not part of the face!"
            )
        if face.vertex_degree() < 4:
            raise ValueError(
                f"Cannot split face with vertex degree <4 (vertex_degree == {face.vertex_degree()})"
            )

        # Find previous and next edges for the splitting vertices
        vx0_next = vx0_prev = vx1_next = vx1_prev = None
        for he in face.adjacent_halfedges():
            if he.src == vx0:
                vx0_next = he
            elif he.next_edge.src == vx0:
                vx0_prev = he
            elif he.src == vx1:
                vx1_next = he
            elif he.next_edge.src == vx1:
                vx1_prev = he

        # If for some reason any of the prev/next halfedges are None, error out
        for he in [vx0_prev, vx0_next, vx1_prev, vx1_next]:
            if he is None:
                raise ValueError(
                    "Error finding previous and next halfedges on the splitting vertices!"
                )

        # Create half edges to split the face
        vx0_vx1 = self._new_halfedge(vx0, vx1, None)
        vx1_vx0 = self._new_halfedge(vx1, vx0, None)

        # Assign twinship
        vx0_vx1.twin = vx1_vx0
        vx1_vx0.twin = vx0_vx1

        # Link up edges
        vx0_vx1.next_edge = vx1_next
        vx1_prev.next_edge = vx1_vx0
        vx1_vx0.next_edge = vx0_next
        vx0_prev.next_edge = vx0_vx1

        # Create the new face and assign it to one of the new half edges
        new_face = self._new_face()
        vx0_vx1.face = new_face
        new_face.halfedge = vx0_vx1

        # Assign the other half edge to the old face and make it the representative halfedge for that face (in case we just assigned the old representative edge to the new face)
        vx1_vx0.face = face
        face.halfedge = vx1_vx0

        # Assign new face to relevant half edges
        for he in traverse_edges(vx0_vx1):
            he.face = new_face

        return face, new_face

    def split_edge(self, he_id):

        # Reference the edge
        he = self.halfedge(he_id)

        # Get source and targets
        src = he.src
        tgt = he.next_edge.src

        # Create new vertex at midpoint of the edge
        xyz = (src.XYZ + tgt.XYZ) / 2
        vx = self._new_vertex(xyz)

        # Create two new halfedges
        vx_tgt = self._new_halfedge(vx, tgt, he.face)
        vx_src = self._new_halfedge(vx, src, he.twin.face)

        # Set the outgoing halfedge for the new vertex
        vx.halfedge = vx_tgt

        # Old next edges
        he_next = he.next_edge
        twin_next = he.twin.next_edge

        # Link old halfedges
        he.next_edge = vx_tgt
        he.twin.next_edge = vx_src

        # Link new halfedges
        vx_tgt.next_edge = he_next
        vx_src.next_edge = twin_next

        # Assign twinship
        vx_tgt.twin = he.twin
        he.twin.twin = vx_tgt
        vx_src.twin = he
        he.twin = vx_src

    def collapse_edge(self, he_id):
        pass

    def flip_edge(self):
        pass

    # ===============================================================
    # Algorithms
    # ===============================================================
    def find_boundary_loops(self):

        # List of boundary loops to return
        boundary_loops = []

        # Get the boundary halfedges and store them in a set for fast reference
        boundary_he = {
            he_id
            for he_id, he in self.halfedges.items() if he.is_boundary()
        }

        # Find the loops
        while len(boundary_he) > 0:
            # Grab a boundary half edge
            first_he = self.halfedge(boundary_he.pop())
            cur_he = first_he

            # Create a boundary loop and add it to the set
            boundary_loops.append(BoundaryLoop(first_he))

            # Traverse the boundary half edges until you get back to the beginning
            while True:
                # Next edge
                next_he = cur_he.next_edge

                # Break on full loop
                if next_he == first_he:
                    break

                # Remove the next edge from the set
                boundary_he.discard(next_he.id)

                # Move to next edge
                cur_he = next_he

        return boundary_loops

    def dual_mesh(self):
        """
        All faces become vertices and face adjacency becomes edges
        """
        pass

    def _earness_test(self, starting_halfedge):

        # Classify vertices into all, reflex, convex, and ears
        reflex = set()
        convex = set()
        ears = set()

        # Iterate over the halfedges
        num_unique_edges = 0
        for he in traverse_edges(starting_halfedge):
            num_unique_edges += 1

            # Determine vertex type
            angle = he.angle()
            if angle >= np.pi:
                reflex.add(he.next_edge.src)
            else:
                convex.add(he.next_edge.src)

        # Create set of ear vertices
        # An ear of a polygon is a triangle formed by three consecutive vertices Vi0, Vi1, and Vi2 for which:
        #   * Vi1 is a convex vertex
        #   * the line segment from Vi0 to Vi2 lies completely inside the polygon
        #   * no vertices of the polygon other than the three vertices of the
        #     triangle are contained in the triangle
        for he0, he1 in pairs(traverse_edges(starting_halfedge)):

            # Reference the vertices
            v0 = he0.src
            v1 = he1.src
            v2 = he1.next_edge.src

            # The ear candidate must be convex
            if v1 not in convex:
                continue

            # Check the reflex vertices to see if they fall within the ear
            is_ear = True
            for rvx in reflex:

                # Exclude the vertices part of this face candidate
                if rvx in (v0, v1, v2):
                    continue

                # Check if the reflex vertex falls within this ear. If so, this can't be an ear
                alpha, beta, gamma = compute_barycentric_coordinates(
                    v0.XYZ, v1.XYZ, v2.XYZ, rvx.XYZ)
                if (0 <= alpha <= 1) \
                    and (0 <= beta <= 1) \
                    and (0 <= gamma <= 1):
                    is_ear = False

            # If none of the reflex vertices fall within this triangle, we have an ear vertex at v1
            if is_ear:
                ears.add((v0, v1, v2))

        return num_unique_edges, ears, reflex, convex

    def triangulate(self, face):

        # Iterate until the face is completely triangulated
        while True:
            # Update the ears, reflex, and convex list
            num_unique_edges, ears, reflex, convex = self._earness_test(
                face.halfedge)

            # Reference the vertices
            v0, v1, v2 = ears.pop()

            # Split the face between v0 and v2, returning the face and the new face.
            face0, face1 = self.split_face(face.id, v0.id, v2.id)

            # Update the face pointer to the face that does not have vertex degree of 3
            face = face0 if face0.vertex_degree() > 3 else face1

            # If this loop had only 3 edges, we filled the last triangle
            if num_unique_edges == 4:
                break

    def fill_hole(self, boundary_loop):
        """
        Implementation of the ear-clipping algorithm for triangular meshes, returning the faces

        https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
        """

        # ====================================================
        # Iteratively create faces from ears
        # ====================================================
        while True:
            # Update the ears, reflex, and convex list
            num_unique_edges, ears, reflex, convex = self._earness_test(
                boundary_loop.halfedge)

            # Reference the vertices
            v0, v1, v2 = ears.pop()

            # Add the new face
            face = self.add_face([v0.id, v1.id, v2.id])

            # Update the boundary loop halfedge to point to the new boundary edge created by adding the new face. If no new boundary halfedge is created, we've closed the hole. This step prevents us from connecting the representative halfedge to a face and thus disassociating the boundary loop with actual boundary halfedges
            for he in face.adjacent_halfedges():
                if he.twin.is_boundary():
                    boundary_loop.halfedge = he.twin

            # If no new boundary is found, set the boundary loop halfedge to be None, indicating an empty loop and break the do-while
            if num_unique_edges == 3:
                boundary_loop.halfedge = None
                break


if __name__ == '__main__':
    vertices = np.array(
        [1.0, 0, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0,
         -1]).reshape(-1, 3)
    faces = np.array([
        # 0, 1, 2, 0, 2, 4, 0, 4, 5, 0, 5, 1, 3, 1, 5, 3, 5, 4, 3, 4, 2, 3, 2, 1
        0,
        2,
        4,
        0,
        4,
        5,
        0,
        5,
        1,
        3,
        1,
        5,
        3,
        5,
        4,
        3,
        4,
        2,
        3,
        2,
        1
    ]).reshape(-1, 3)
    hemesh = HalfEdgeMesh(vertices, faces)
    # hemesh.add_face([0, 1, 2])
    for f in hemesh.faces_adjacent_to_face(2):
        print(f)
    import ipdb
    ipdb.set_trace()
    loop = hemesh.find_boundary_loops()
    hemesh.write_mesh("hemesh_output_hole.ply")
    # hemesh.write_mesh("hemesh_output.ply")
