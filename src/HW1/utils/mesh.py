from abc import ABC
from src.HW1.utils.io import read_off

import numpy as np
import pyvista as pv
import scipy.sparse as sparse


class Mesh(ABC):
    """
    A class describing a mesh via its vertices & faces
    """

    def __init__(self, mesh_file: str = None):
        """
        Initialize the mesh from a file

        :param mesh_file: (str) Path to the '.off' file from which to load the mesh.
        """

        # In order to allow instantiating the Mesh class without supplying a file
        if mesh_file is not None:
            data = read_off(mesh_file)

            self.v = data[0]
            self.f = data[1]

        self.Avf = None
        self.Avv = None
        self.vertices_degree = None

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
            self.Avv = (Avf @ Avf.T) == 2

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
        assert scalar_func in ('degree', 'coo'), \
            "'scalar_func' must be either 'degree' or 'coo'"

        vertices = self._get_vertices_array()
        faces = self._get_faces_array()

        colors = None
        if scalar_func == 'inds':
            colors = np.sum(faces, 1)

        elif scalar_func == 'coo':
            colors = np.sqrt(np.sum(np.power(vertices, 2), 1))

        mesh = pv.PolyData(vertices, faces)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style='surface', cmap='hot',
                         scalars=colors)
        plotter.show()

    def faces_normals(self, unit_norm: bool = True) -> np.ndarray:
        """
        A method for computing the 'outward-facing' normal vectors of each face in
        the mesh.

        :param unit_norm: (bool) Whether to normalize the computed normals to have an
        L2 norm of 1.

        :return: (np.ndarray) A NumPy array of shape (|F|, 3), containing the
        computed normals
        """

        # Get the face adjacency matrix
        Afv = np.array(self.vertex_face_adjacency().todense())
        n_faces = len(self.f)

        # For each face, get two connected vertices
        vertices_inds = [np.where(Afv[:, f] == 1)[0] for f in range(n_faces)]

        # Compute the normal vector for each face using the detected vertices
        normals = np.concatenate([np.expand_dims(np.cross(
            (np.array(self.v[vertices[1]]) - np.array(self.v[vertices[0]])),
            (np.array(self.v[vertices[2]]) - np.array(self.v[vertices[0]])),
        ), 0)
            for vertices in vertices_inds], 0).astype(np.float)

        if unit_norm:
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            normals /= norms

        normals = np.abs(normals)

        return normals
