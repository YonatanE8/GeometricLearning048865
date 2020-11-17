from abc import ABC
from src.HW1.utils.io import read_off

import numpy as np
import pyvista as pv
import scipy.sparse as sparse


class Mesh(ABC):
    """
    A class describing a mesh via its vertices & faces
    """

    def __init__(self, mesh_file: str = None, normals_unit_norm: bool = True):
        """
        Initialize the mesh from a file
        :param mesh_file: (str) Path to the '.off' file from which to load the mesh.
        :param normals_unit_norm: (bool) Whether to normalize the computed normals
        to have an L2 norm of 1.
        """

        # In order to allow instantiating the Mesh class without supplying a file
        if mesh_file is not None:
            data = read_off(mesh_file)

            self.v = data[0]
            self.f = data[1]

        self.unit_norm = normals_unit_norm

        self.Avf = None
        self.Avv = None
        self.vertices_degree = None

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """

        :param vector:
        :return:
        """

        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        vector = vector / norms

        return vector

    @staticmethod
    def _calc_vertex_angle(main_vertex: np.ndarray,
                           vertices: np.ndarray, main_vertex_ind: int) -> float:
        """

        :param main_vertex:
        :param vertices:
        :param main_vertex_ind:
        :return:
        """

        other_vertices = np.setdiff1d(vertices, np.array([main_vertex_ind, ]))
        vertex_1 = vertices[other_vertices[0]]
        vertex_2 = vertices[other_vertices[1]]

        a = np.expand_dims(np.array(main_vertex - vertex_1), 0)
        b = np.expand_dims(np.array(main_vertex - vertex_2), 1)

        angle = np.arccos(((np.matmul(a, b).item())
                           / (np.linalg.norm(a) * np.linalg.norm(b))))

        return angle

    def _get_vertices_array(self):
        """
        Utility method for gathering all vertices into a single NumPy array
        """

        vertices = np.array(self.v)

        return vertices

    def _get_faces_array(self):
        """
        Utility method for gathering all faces into a single NumPy array
        """

        faces = np.array(self.f)
        faces = np.concatenate(
            (
                np.expand_dims(len(self.f[0]) * np.ones((len(self.f)), ),
                               1).astype(np.int),
                faces
            ), 1
        )

        return faces

    def vertex_face_adjacency(self) -> sparse.coo_matrix:
        """
        This method computes a boolean vertex - face adjacency matrix 'A',
        describing the one ring neighbors of each vertice,
        where $A \in \set{R}^{|V| X |F|}$ and $|V| = # Vertices$, $|F| = # faces$.

        :return: (scipy.sparse.coo_matrix) |V| X |F| Adjacency matrix describing the
         vertex - face adjacency matrix.
        """

        if self.Avf is None:
            n_vertices = len(self.v)
            n_faces = len(self.f)

            adjacencies = np.concatenate([
                [np.array([i in self.f[j] for j in range(n_faces)])
                 for i in range(n_vertices)]], 0)

            self.Avf = sparse.coo_matrix(adjacencies,
                                         shape=[n_vertices, n_faces],
                                         dtype=np.bool)

        return self.Avf

    def vertex_vertex_adjacency(self) -> sparse.coo_matrix:
        """
        This method computes a boolean vertex - vertex adjacency matrix 'A',
        describing connected vertices of each vertice,
        where $A \in \set{R}^{|V| X |V|}$ and $|V| = # Vertices$.

        :return: (scipy.sparse.coo_matrix) |V| X |V| Adjacency matrix describing the
         vertex - vertex adjacency matrix.
        """

        if self.Avv is None:
            # Compute the vertex - vertex adjacency matrix
            Avf = self.vertex_face_adjacency().astype(np.int)

            # Note that 2 vertices are adjacent if they share 2 faces
            Avf_sq = Avf @ Avf.T
            inds_1 = Avf_sq <= 2
            inds_2 = Avf_sq > 0
            Avv = np.array((inds_1 - inds_2).todense()) == False
            self.Avv = sparse.coo_matrix(Avv,
                                         shape=Avv.shape,
                                         dtype=np.bool)

        return self.Avv

    def vertex_degree(self) -> np.ndarray:
        """
        This method computes the vertex-degree for all vertices.

        :return: (np.ndarray) NumPy vector of shape (|V|, ) where each entry i denotes
        the degree of vertex i.
        """

        if self.vertices_degree is None:
            Avv = self.vertex_vertex_adjacency().sum(1)
            self.vertices_degree = np.array(Avv.sum(1)).squeeze()

        return self.vertices_degree

    def render_wireframe(self) -> None:
        """
        Render the mesh's vertices in wireframe view using the PyVista package

        :return: None
        """

        vertices = self._get_vertices_array()
        faces = self._get_faces_array()
        mesh = pv.PolyData(vertices, faces)

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style='wireframe')
        plotter.show()

    def render_pointcloud(self, scalar_func: str = 'degree') -> None:
        """
        Render the mesh's vertices in point cloud view using the PyVista package

        :param scalar_func: (str) A function $f: \set{R}^{3} -> \set{R}$
        indicating the scalar value to use as color for the point-cloud visualization.
        Options are: 'degree' - uses the vertex's degree,
        and 'coo' - uses the coordinates RMS value. Defaults to 'degree'.
        :return:
        """

        # Validate input
        assert scalar_func in ('degree', 'coo'), \
            "'scalar_func' must be either 'degree' or 'coo'"

        vertices = self._get_vertices_array()
        faces = self._get_faces_array()

        colors = None
        if scalar_func == 'degree':
            colors = np.sum(np.array(self.vertex_vertex_adjacency().todense()), 1)

        elif scalar_func == 'coo':
            colors = np.sqrt(np.sum(np.power(vertices, 2), 1))

        mesh = pv.PolyData(vertices, faces)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style='points', cmap='hot',
                         render_points_as_spheres=True, scalars=colors)
        plotter.show()

    def render_surface(self, scalar_func: str = 'inds') -> None:
        """
        Render the mesh surface using the PyVista package

        :param scalar_func: (str) A function $f: \set{R}^{3} -> \set{R}$
        indicating the scalar value to use as color for the point-cloud visualization.
        Options are: 'degree' - uses the vertex's degree,
        and 'coo' - uses the coordinates RMS value. Defaults to 'degree'.
        :return:
        """

        # Validate input
        assert scalar_func in ('inds', 'degree', 'coo'), \
            "'scalar_func' must be either 'inds', 'degree' or 'coo'"

        vertices = self._get_vertices_array()
        faces = self._get_faces_array()

        colors = None
        if scalar_func == 'inds':
            colors = np.sum(faces, 1)

        elif scalar_func == 'degree':
            colors = np.sum(np.array(self.vertex_vertex_adjacency().todense()), 1)

        elif scalar_func == 'coo':
            colors = np.sqrt(np.sum(np.power(vertices, 2), 1))

        # If using vertex-coloring function, interpolate colors for all faces
        if scalar_func in ('degree', 'coo'):
            colors = np.array([
                np.mean(colors[faces[f, 1:]]) for f in range(faces.shape[0])
            ])

        mesh = pv.PolyData(vertices, faces)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style='surface', cmap='hot',
                         scalars=colors)
        plotter.show()

    def _faces_normals(self) -> np.ndarray:
        """
        A utility method for computing the 'outward-facing' normal vectors of each face in
        the mesh.

        :return: (np.ndarray) A NumPy array of shape (|F|, 3), containing the
        computed normals
        """

        # Get all faces & vertices
        v = np.array(self.v)
        f = np.array(self.f)

        # Compute the normal vector for each face using the detected vertices
        a = v[f[:, 0], :]
        b = v[f[:, 1], :]
        c = v[f[:, 2], :]
        normals = np.cross((b - a), (c - a))

        if self.unit_norm:
            normals = self._normalize_vector(normals)

        return normals

    @property
    def normals(self) -> np.ndarray:
        """
        A mesh property, containing the normals of each face in the mesh

        :return: (np.ndarray) A NumPy array of shape (|F|, 3), containing the
        computed normals
        """

        return self._faces_normals()

    def _faces_barycenters(self) -> np.ndarray:
        """
        A utility method for computing the 'barycenter' of each face, i.e. the mean
        of the vertices composing each face

        :return: (np.ndarray) A NumPy array of shape (|F|, ), containing the
        computed barycenters
        """

        # Get the coordinates of all vertices for each face
        faces = np.array(self.f)
        faces_vertices = np.array(self.v)
        faces_vertices = np.array([
            np.expand_dims([faces_vertices[faces[f, :]]], 0)
            for f in range(faces.shape[0])
        ])
        faces_vertices = np.concatenate(faces_vertices, 0)

        # Compute the barycenters
        barycenters = np.mean(faces_vertices, 1)

        return barycenters

    @property
    def barycenters(self) -> np.ndarray:
        """
        A mesh property, containing the barycenters of each face in the mesh

        :return: (np.ndarray) A NumPy array of shape (|F|, ), containing the
        computed barycenters
        """

        return self._faces_barycenters()

    def _faces_areas(self) -> np.ndarray:
        """
        A utility method for computing the area of each face using Heron's formula

        :return: (np.ndarray) A NumPy array of shape (|F|, ), containing the
        computed areas
        """

        # Get the coordinates of all vertices for each face
        vertices = np.array(self.v)
        faces = np.array(self.f)
        faces_vertices = [
            (
                vertices[faces[f][0], :],
                vertices[faces[f][1], :],
                vertices[faces[f][2], :],
            )
            for f in range(faces.shape[0])]
        triangles = np.concatenate([
            np.expand_dims(np.array((
                np.sqrt(np.power((tri[0] - tri[1]), 2)),
                np.sqrt(np.power((tri[0] - tri[2]), 2)),
                np.sqrt(np.power((tri[1] - tri[2]), 2)),
            )), 0)
            for tri in faces_vertices
        ], 0)

        # Compute the s term from Heron's formula for all triangles
        s = np.sum(triangles, 1) / 2

        # Compute the area of each face
        areas = np.array([
            np.sqrt(
                (s[t] *
                 (s[t] - triangles[t, 0]) *
                 (s[t] - triangles[t, 1]) *
                 (s[t] - triangles[t, 2]))
            )
            for t, tri in enumerate(triangles)
            ])

        return areas

    @property
    def areas(self) -> np.ndarray:
        """
        A mesh property, containing the area of each face in the mesh

        :return: (np.ndarray) A NumPy array of shape (|F|, ), containing the
        computed areas
        """

        return self._faces_areas()

    def _vertices_barycenters_areas(self) -> np.ndarray:
        """
        A utility method for computing the barycenter area of each vertex in the mesh

        :return: (np.ndarray) A NumPy array of shape (|V|, ), containing the
        computed barycenter areas
        """

        # Get the vertex-face adjacency matrix
        n_vertices = len(self.v)
        Avf = np.array(self.vertex_face_adjacency().todense())

        # Get faces matrix
        faces = self._get_faces_array()

        # Get the faces areas matrix
        faces_areas = self.areas

        # Compute the barycenter area for each vertex based on all adjacent faces
        # which are triangular
        areas = np.array([
            ((1 / 3) * np.sum([faces_areas[face_ind]
                               for face_ind in Avf[i, :] if faces[face_ind, 0] == 3]))
            for i in range(n_vertices)
        ])

        return areas

    @property
    def barycenters_areas(self) -> np.ndarray:
        """
        A mesh property, containing the barycenter area of each vertex in the mesh

        :return: (np.ndarray) A NumPy array of shape (|V|, ), containing the
        computed barycenter areas
        """

        return self._vertices_barycenters_areas()

    def _compute_vertex_normals(self) -> np.ndarray:
        """

        :return:
        """

        # Get faces areas
        Af = self.areas

        # Get faces normals
        normals = self.normals

        # Get vertices-faces adjacency matrix
        Avf = np.array(self.vertex_face_adjacency().todense().astype(np.int))

        # Sum all weighted areas_per_vertex
        if self.unit_norm:
            vertices_areas = np.matmul(Avf, (np.expand_dims(Af, 1)
                                             * self._normalize_vector(normals)))

        else:
            vertices_areas = np.matmul(Avf, (np.expand_dims(Af, 1) * normals))

        return vertices_areas

    @property
    def vertex_normals(self) -> np.ndarray:
        """

        :return:
        """

        return self._compute_vertex_normals()

    def _compute_gaussian_curvature(self) -> np.ndarray:
        """

        :return:
        """

        # For each vertex, get all adjacent faces and vertices
        Avf = np.array(self.vertex_face_adjacency().todense().astype(np.int))

        n_vertex = Avf.shape[0]
        faces_per_vertex = np.array([
            np.where(Avf[v, :] == 1)[0] for v in range(n_vertex)
        ])

        vertices_per_vertex = np.array([
            [np.where(Avf[:, f] == 1)[0] for f in faces] for faces in faces_per_vertex
        ])

        angels_per_vertex = [
            self._calc_vertex_angle(
                main_vertex=self.v[v], vertices=vertices_per_vertex[v],
                main_vertex_ind=v)
            for v, vertices in enumerate(vertices_per_vertex)
        ]

        # Get all vertices - areas
        Av = self._vertices_barycenters_areas()

        # Calculate final curvatures
        curvatures = (np.array([
            (2 * np.pi - np.sum(angles)) for angles in angels_per_vertex
        ]) / Av)

        return curvatures

    @property
    def gaussian_curvature(self) -> np.ndarray:
        """

        :return:
        """

        return self._compute_gaussian_curvature()



