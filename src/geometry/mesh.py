from abc import ABC
from src.utils.io import read_off

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
    def _normalize_vector(vector: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Utility method for normalizing an input vector such that the L2-norm of the
        normalized vector will be equal to 1

        :param vector: (np.ndarray) The vector to normalize

        :return: (np.ndarray, np.ndarray) Tuple containing the L-2 normalized vector
         in index 0 and the computed norms in index 1
        """

        norms = np.linalg.norm(vector, axis=1, keepdims=True)
        vector = vector / norms

        return vector, norms

    @staticmethod
    def _calc_vertex_angle(main_vertex: np.ndarray,
                           vertices: np.ndarray, main_vertex_ind: int,
                           all_vertices_inds: np.ndarray) -> float:
        """
        Utility method for computing specific vertex's angle in a specific face

        :param main_vertex: (np.ndarray) Coordinates vector of the vertex for
        which the angle is computed
        :param vertices: (np.ndarray) Coordinates of all three vertices
        :param main_vertex_ind: (int) Index of the vertex for which the angle
         is computed
        :param all_vertices_inds: (np.ndarray) Indices of all vertices

        :return: (float) The angle of main_vertex in the face containing all
         three vertices
        """

        # Computing the angle using the cosine theorem:
        # theta = arccos((a^2 + b^2 - c^2) / (2ab))
        other_vertices = np.setdiff1d(np.arange(3),
                                      np.where(all_vertices_inds == main_vertex_ind)[0])
        vertex_1 = vertices[other_vertices[0]]
        vertex_2 = vertices[other_vertices[1]]

        a = vertex_1 - main_vertex
        b = vertex_2 - main_vertex
        c = vertex_2 - vertex_1
        size_a = np.sqrt(np.sum(np.power(a, 2)))
        size_b = np.sqrt(np.sum(np.power(b, 2)))
        size_c = np.sqrt(np.sum(np.power(c, 2)))

        angle = np.arccos((((size_a ** 2) + (size_b ** 2) - (size_c ** 2)) /
                           (2 * size_a * size_b)))

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
        assert scalar_func in ('inds', 'degree', 'coo', 'face_area', 'vertex_area'), \
            "'scalar_func' must be either 'inds', 'degree', 'coo', " \
            "'face_area' or 'vertex_area'"

        vertices = self._get_vertices_array()
        faces = self._get_faces_array()

        colors = None
        if scalar_func == 'inds':
            colors = np.sum(faces, 1)

        elif scalar_func == 'degree':
            colors = np.sum(np.array(self.vertex_vertex_adjacency().todense()), 1)

        elif scalar_func == 'coo':
            colors = np.sqrt(np.sum(np.power(vertices, 2), 1))

        elif scalar_func == 'face_area':
            colors = self.areas

        # If using vertex-coloring function, interpolate colors for all faces
        if scalar_func in ('degree', 'coo'):
            colors = np.array([
                np.mean(colors[faces[f, 1:]]) for f in range(faces.shape[0])
            ])

        elif scalar_func == 'vertex_area':
            colors = self.barycenters_areas
            colors = np.array([
                np.mean(colors[faces[f, 1:]]) for f in range(faces.shape[0])
            ])

        mesh = pv.PolyData(vertices, faces)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, style='surface', cmap='hot',
                         scalars=colors)
        plotter.show()

    def _faces_normals(self) -> (np.ndarray, np.ndarray):
        """
        A utility method for computing the 'outward-facing' normal vectors of each face in
        the mesh.

        :return: (np.ndarray, np.ndarray) Tuple containing an array of shape (|F|, 3),
        containing the computed normals, and a np.ndarray containing the computed norms.
         If the normals are not normalize then the norms are an empty ndarray.
        """

        # Get all faces & vertices
        v = np.array(self.v)
        f = np.array(self.f)

        # Compute the normal vector for each face using the detected vertices
        a = v[f[:, 0], :]
        b = v[f[:, 1], :]
        c = v[f[:, 2], :]
        normals = np.cross((b - a), (c - a))

        norms = np.array([])
        if self.unit_norm:
            normals, norms = self._normalize_vector(normals)

        return normals, norms

    @property
    def normals(self) -> np.ndarray:
        """
        A mesh property, containing the normals of each face in the mesh

        :return: (np.ndarray) A NumPy array of shape (|F|, 3), containing the
        computed normals
        """

        return self._faces_normals()[0]

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

        # Compute the barycenters
        barycenters = np.array(
            [np.mean(np.concatenate([np.expand_dims(np.array(faces_vertices[v]), 0)
                                     for v in face], 0), 0)
             for face in faces]
        )

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
                np.sqrt(np.sum(np.power((tri[0] - tri[1]), 2))),
                np.sqrt(np.sum(np.power((tri[0] - tri[2]), 2))),
                np.sqrt(np.sum(np.power((tri[1] - tri[2]), 2))),
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
        Avf = self.vertex_face_adjacency()

        # Get the faces areas matrix
        faces_areas = self.areas

        # Note that the we can get the Barycenters areas simply by
        # 1/3 * Avf * faces_areas as the adjacency matrix will automatically sum over
        # all relevant faces for each vertex
        areas = (1 / 3) * Avf * faces_areas

        return areas

    @property
    def barycenters_areas(self) -> np.ndarray:
        """
        A mesh property, containing the barycenter area of each vertex in the mesh

        :return: (np.ndarray) A NumPy array of shape (|V|, ), containing the
        computed barycenter areas
        """

        return self._vertices_barycenters_areas()

    def _compute_vertex_normals(self) -> (np.ndarray, np.ndarray):
        """
        A utility method for computing the vertices normals vectors.

        :return: (np.ndarray) Tuple containing an array of shape (|V|, 3), with the
        normal of each vertex, if the unit_norm attribute is set to True,
        each normal will have a L2-norm of 1. and a np.ndarray containing the computed
        norms. If the normals are not normalize then the norms are an empty ndarray.
        """

        # Get faces areas
        Af = self.areas

        # Get faces normals, if normalization is required then we need to do it only
        # after computing the final vertex normals
        normalize = False
        if self.unit_norm:
            self.unit_norm = False
            normalize = True

        normals = self.normals

        # Get vertices-faces adjacency matrix
        Avf = self.vertex_face_adjacency()

        # Calculate all weighted normals
        weighted_normals = np.expand_dims(Af, 1) * normals

        # Compute all weighted areas_per_vertex
        vertices_normals = Avf @ weighted_normals

        # Normalize normals to have unit-norm is required
        norms = np.array([])
        if normalize:
            self.unit_norm = True
            norms = np.linalg.norm(vertices_normals, axis=1)
            vertices_normals = vertices_normals / np.expand_dims(norms, 1)

        return vertices_normals, norms

    @property
    def vertex_normals(self) -> np.ndarray:
        """
        A mesh property, containing the normal vector of each vertex in the mesh.

        :return: (np.ndarray) A NumPy array of shape (|V|, 3), containing the
        normal of each vertex, if the unit_norm attribute is set to True,
        each normal will have a L2-norm of 1.
        """

        return self._compute_vertex_normals()[0]

    def _euler_characteristic(self) -> int:
        """
        A utility method for computing the Euler Characteristic $\chi$ of the mesh.

        :return: (int) The Euler Characteristic $\chi$ of the mesh.
        """

        # Get V and F immediately
        n_v = len(self.v)
        n_f = len(self.f)

        # E is the # of non-zero entries in the upper-triangle of the
        # vertex-vertex adjacency matrix.
        n_e = self.vertex_vertex_adjacency().todense()
        n_e = int(np.sum([np.sum(n_e[r, r:]) for r in range(n_v)]).item())

        return n_v - n_e + n_f

    @property
    def euler_characteristic(self) -> int:
        """
        A mesh property, containing the Euler Characteristic $\chi$ of the mesh.

        :return: (int) The Euler Characteristic $\chi$ of the mesh.
        """

        return self._euler_characteristic()

    def _compute_gaussian_curvature(self) -> np.ndarray:
        """
        A utility method for computing the Gaussian curvature for each vertex.

        :return: (np.ndarray) A NumPy array of shape (|V|, ), containing the
        Gaussian curvature of each vertex.
        """

        # For each vertex, get all adjacent faces and vertices
        Avf = np.array(self.vertex_face_adjacency().todense().astype(np.int))

        # Pre-computations
        n_vertex = len(self.v)
        faces_per_vertex = np.array([
            np.where(Avf[v, :] == 1)[0] for v in range(n_vertex)
        ])
        vertices_per_vertex = [
            [np.where(Avf[:, f] == 1)[0] for f in faces]
            for faces in faces_per_vertex
        ]
        vertices_array = self._get_vertices_array()

        # Compute the angle per face per vertex
        angels_per_vertex = [
            np.array([self._calc_vertex_angle(
                main_vertex=self.v[v], vertices=vertices_array[f],
                main_vertex_ind=v, all_vertices_inds=f)
                for f in vertices_per_vertex[v]])
            for v, vertices in enumerate(vertices_per_vertex)
        ]

        # Get all vertices' areas
        Av = self._vertices_barycenters_areas()

        # Calculate final curvatures
        curvatures = ((2 * np.pi) - np.array([np.sum(angles)
                                              for angles in angels_per_vertex])) / Av

        return curvatures

    @property
    def gaussian_curvature(self) -> np.ndarray:
        """
        A mesh property, containing the Gaussian curvature for each vertex.

        :return: (np.ndarray) A NumPy array of shape (|V|, ), containing the
        Gaussian curvature of each vertex.
        """

        return self._compute_gaussian_curvature()

    def render_vertices_normals(self, normalize: bool, add_norms: bool = False) -> None:
        """
        A method for visualizing the normal vectors of all vertices in the mesh.

        :param normalize: (bool) Whether to normalize the normal vectors to
        have L2-norm = 1. Note that this is irrespective of whether the Mesh class was
        instantiated with normals_unit_norm = True or not.
        :param add_norms: (bool) Whether to also visualize the norm of each
        un-normalized vertex's normal.

        :return: None
        """

        vertices = self._get_vertices_array()
        faces = self._get_faces_array()
        mesh = pv.PolyData(vertices, faces)
        mesh.vectors = vertices

        if add_norms:
            arrows = mesh.glyph(orient="Normals", tolerance=0.05)

            colors = self._compute_vertex_normals()[1]
            colors = np.array([
                np.mean(colors[faces[f, 1:]]) for f in range(faces.shape[0])
            ])

            plotter = pv.Plotter()
            plotter.add_mesh(arrows, color="black", lighting=False)
            plotter.add_mesh(mesh, style='surface', cmap='hot', scalars=colors)
            plotter.show()

        else:
            if normalize:
                arrows = mesh.glyph(scale="Normals", orient="Normals", tolerance=0.05)

            else:
                arrows = mesh.glyph(orient="Normals", tolerance=0.05)

            plotter = pv.Plotter()
            plotter.add_mesh(arrows, color="black", lighting=False)
            plotter.add_mesh(mesh, style="wireframe")
            plotter.show()

    def render_faces_normals(self, normalize: bool, add_norms: bool = False) -> None:
        """
        A method for visualizing the normal vectors of all vertices in the mesh.

        :param normalize: (bool) Whether to normalize the normal vectors to
        have L2-norm = 1. Note that this is irrespective of whether the Mesh class was
        instantiated with normals_unit_norm = True or not.
        :param add_norms: (bool) Whether to also visualize the norm of each
        un-normalized vertex's normal.

        :return: None
        """

        vertices = self._get_vertices_array()
        faces = self._get_faces_array()
        mesh = pv.PolyData(vertices, faces)
        mesh.vectors = self.barycenters

        if add_norms:
            arrows = mesh.glyph(orient="Normals", tolerance=0.05)
            colors = self._faces_normals()[1]

            plotter = pv.Plotter()
            plotter.add_mesh(arrows, color="black", lighting=False)
            plotter.add_mesh(mesh, style='surface', cmap='hot', scalars=colors)
            plotter.show()

        else:
            if normalize:
                arrows = mesh.glyph(scale="Normals", orient="Normals", tolerance=0.05)

            else:
                arrows = mesh.glyph(orient="Normals", tolerance=0.05)

            plotter = pv.Plotter()
            plotter.add_mesh(arrows, color="black", lighting=False)
            plotter.add_mesh(mesh, style='surface', cmap='hot')
            plotter.show()

    def _compute_vertex_centroid(self) -> np.ndarray:
        """
        Utility method for computing the vertices centroid of the mesh.

        :return: (np.ndarray) Coordinates of the mesh's vertices centroid
        """

        vertices = self._get_vertices_array()
        centroid = np.mean(vertices, 0)

        return centroid

    @property
    def vertices_centroid(self) -> np.ndarray:
        """
        A mesh property containing the mesh's vertices centroid.

        :return: (np.ndarray) Coordinates of the mesh's vertices centroid
        """

        return self._compute_vertex_centroid()



