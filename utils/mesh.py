from abc import ABC
from HW1.utils.io import read_off

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

    def vertex_face_adjacency(self):
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

    def vertex_vertex_adjacency(self):
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

    def vertex_degree(self):
        """
        This method computes the vertex-degree for all vertices.

        :return: (np.ndarray) NumPy vector of shape (|V|, ) where each entry i denotes
        the degree of vertex i.
        """

        if self.vertices_degree is None:
            Avv = self.vertex_vertex_adjacency().sum(1)
            self.vertices_degree = np.array(Avv.sum(1)).squeeze()

        return self.vertices_degree

    def render_wireframe(self):
        """
        Render the mesh's vertices in wireframe view using the PyVista package

        :return: None
        """

        x = np.expand_dims(np.array([v[0] for v in self.v]), 1)
        y = np.expand_dims(np.array([v[1] for v in self.v]), 1)
        z = np.expand_dims(np.array([v[2] for v in self.v]), 1)
        vertices = np.concatenate((x, y, z), 1)
        faces = np.concatenate([np.expand_dims(f, 1) for f in self.f], 1)
        mesh = pv.PolyData(vertices, faces)

        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style='points')
        plotter.show()




