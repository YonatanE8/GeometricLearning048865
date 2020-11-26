from src.utils.io import read_off
from src.geometry.mesh import Mesh
from src import PROJECT_ROOT

import os
import glob
import pytest
import numpy as np


class TestMesh:
    @pytest.fixture()
    def vertices_setup(self):
        self.vertices = [
            [0, 1, 0],
            [0, 1, 0.5],
            [0, 1, 1],
            [0.5, 1, 1],
            [1, 1, 1],
            [1, 1, 0.5],
            [1, 1, 0],
            [0.5, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0.5, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
        ]

        self.faces = [
            [0, 3, 6],
            [0, 2, 3],
            [3, 4, 6],
            [8, 1, 0],
            [9, 1, 8],
            [9, 2, 1],
            [3, 2, 9],
            [11, 3, 9],
            [11, 4, 3],
            [4, 11, 5],
            [12, 5, 11],
            [5, 12, 6],
            [12, 7, 6],
            [12, 8, 7],
            [8, 0, 7],
            [10, 9, 8],
            [12, 10, 8],
            [12, 11, 10],
        ]

    @pytest.fixture()
    def simple_cube(self):
        self.vertices = [
            [0, 1, 0],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 0, 0],
        ]

        self.faces = [
            [0, 1, 2, 3],
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [6, 7, 3, 2],
            [7, 4, 0, 3],
            [4, 5, 6, 7],
        ]

    def test_mesh_init(self):
        data_dir = os.path.join(PROJECT_ROOT, 'data', 'off_files')
        file = glob.glob(os.path.join(data_dir, '*.off'))[0]

        # Load a mesh using read_off and using the Mesh class and assert that they
        # are identical
        data = read_off(file)
        data_v = data[0]
        data_f = data[1]

        mesh = Mesh(file)
        mesh_v = mesh.v
        mesh_f = mesh.f

        assert len(data_v) == len(mesh_v)
        assert len(data_f) == len(mesh_f)

        vertices_diffs = [np.sum((np.array(data_v[v]) - np.array(mesh_v[v])))
                          for v in range(len(data_v))]
        faces_diffs = [np.sum((np.array(data_f[f]) - np.array(mesh_f[f])))
                       for f in range(len(data_f))]

        assert np.sum(vertices_diffs) == 0.
        assert np.sum(faces_diffs) == 0

    def test_vertex_face_adjacency(self, vertices_setup):
        # Test the method on a small scale problem
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces

        a = mesh.vertex_face_adjacency()

        assert a.shape == (len(self.vertices), len(self.faces))

        cols = np.array([0, 1, 3, 14, 3, 4, 5, 1, 5, 6, 0, 1, 2, 6, 7, 8, 2,
                         8, 9, 9, 10, 11, 0, 2, 11, 12, 12, 13, 14, 3, 4, 13, 14, 15,
                         16, 4, 5, 6, 7, 15, 15, 16, 17, 7, 8, 9, 10, 17, 10, 11, 12,
                         13, 16, 17], dtype=np.int32)
        rows = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4,
                         4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 8,
                         8, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12,
                         12, 12, 12], dtype=np.int32)

        assert all(a.col == cols)
        assert all(a.row == rows)

    def test_vertex_vertex_adjacency(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces

        Avv = mesh.vertex_vertex_adjacency()

        assert Avv.shape == (len(self.vertices), len(self.vertices))

        Avv_gt = np.array([
            [False, True, True, True, False, False, True, True, True,
             False, False, False, False],
            [True, False, True, False, False, False, False, False,
             True, True, False, False, False],
            [True, True, False, True, False, False, False, False,
             False, True, False, False, False],
            [True, False, True, False, True, False, True, False,
             False, True, False, True, False],
            [False, False, False, True, False, True, True, False,
             False, False, False, True, False],
            [False, False, False, False, True, False, True, False,
             False, False, False, True, True],
            [True, False, False, True, True, True, False, True,
             False, False, False, False, True],
            [True, False, False, False, False, False, True, False,
             True, False, False, False, True],
            [True, True, False, False, False, False, False, True,
             False, True, True, False, True],
            [False, True, True, True, False, False, False, False,
             True, False, True, True, False],
            [False, False, False, False, False, False, False, False,
             True, True, False, True, True],
            [False, False, False, True, True, True, False, False,
             False, True, True, False, True],
            [False, False, False, False, False, True, True, True,
             True, False, True, True, False],
        ])

        assert np.sum(1 - (np.array(Avv.todense()) ==
                           Avv_gt).reshape((-1,)).astype(np.int)) == 0

        assert np.sum(np.diag(np.array(Avv.todense()).astype(np.int))) == 0

    def test_vertices_degree(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces

        Vd = mesh.vertex_degree()
        Vd_gt = np.array(
            [6, 4, 4, 6, 4, 4, 6, 4, 6, 6, 4, 6, 6]
        )

        assert Vd.shape == (len(self.vertices),)
        assert np.sum(Vd - Vd_gt) == 0

    def test_faces_normals(self, vertices_setup):
        mesh = Mesh(normals_unit_norm=False)
        mesh.v = self.vertices
        mesh.f = self.faces
        normals = mesh.normals

        mesh = Mesh(normals_unit_norm=True)
        mesh.v = self.vertices
        mesh.f = self.faces
        normed_normals = mesh.normals
        norms = np.sqrt(np.sum(np.power(normed_normals, 2), 1)).tolist()

        assert normals.shape == (len(mesh.f), 3)
        assert normed_normals.shape == (len(mesh.f), 3)
        for norm in norms:
            pytest.approx(norm, 1.0, 1e-8)

        directions = np.array([[0., 1., 0.],
                               [0., 0.5, 0.],
                               [-0., 0.5, 0.],
                               [-0.5, 0., 0.],
                               [-1., 0., 0.],
                               [-0.5, 0., 0.],
                               [0., 0., 0.5],
                               [0., 0., 1.],
                               [0., -0., 0.5],
                               [0.5, 0., 0.],
                               [1., 0., 0.],
                               [0.5, 0., 0.],
                               [0., 0., -0.5],
                               [0., 0., -1.],
                               [0., 0., -0.5],
                               [-0., -0.5, 0.],
                               [0., -1., 0.],
                               [0., -0.5, 0.]])

        norm_directions = np.array([[0., 1., 0.],
                                    [0., 1., 0.],
                                    [-0., 1., 0.],
                                    [-1., 0., 0.],
                                    [-1., 0., 0.],
                                    [-1., 0., 0.],
                                    [0., 0., 1.],
                                    [0., 0., 1.],
                                    [0., -0., 1.],
                                    [1., 0., 0.],
                                    [1., 0., 0.],
                                    [1., 0., 0.],
                                    [0., 0., -1.],
                                    [0., 0., -1.],
                                    [0., 0., -1.],
                                    [-0., -1., 0.],
                                    [0., -1., 0.],
                                    [0., -1., 0.]])

        assert np.sum(directions - normals) == 0
        assert np.sum(norm_directions - normed_normals) == 0

    def test_barycenters(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces
        barycenters = mesh.barycenters

        n_faces = len(self.faces)
        gt_barycenters = np.array(
            [np.mean(np.concatenate([np.expand_dims(np.array(self.vertices[v]), 0)
                                     for v in self.faces[f]], 0), 0)
             for f in range(n_faces)]
        )

        assert np.sum((barycenters - gt_barycenters)) == 0

    def test_areas(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces
        areas = mesh.areas

        gt_areas = []
        for face in self.faces:
            v1 = np.array(self.vertices[face[0]])
            v2 = np.array(self.vertices[face[1]])
            v3 = np.array(self.vertices[face[2]])

            a = np.sqrt(np.sum(np.power((v1 - v2), 2)))
            b = np.sqrt(np.sum(np.power((v1 - v3), 2)))
            c = np.sqrt(np.sum(np.power((v2 - v3), 2)))

            s = (a + b + c) / 2
            area = np.sqrt((s * (s - a) * (s - b) * (s - c)))

            gt_areas.append(area)

        gt_areas = np.array(gt_areas)

        diff = np.sum((gt_areas - areas))
        assert pytest.approx(diff, 0, 1e-8)
        for area in areas:
            assert area > 0

    def test_barycenters_areas(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces
        barycenters_areas = mesh.barycenters_areas

        # Get faces areas
        faces_areas = mesh.areas

        gt_vertices_areas = []
        vertex_face_adj = mesh.vertex_face_adjacency().todense().astype(np.int).tolist()
        vertex_face_adj = [np.where(np.array(row) == 1)[0] for row in vertex_face_adj]
        for v in vertex_face_adj:
            gt_vertices_areas.append((np.sum(faces_areas[v]) / 3))

        gt_vertices_areas = np.array(gt_vertices_areas)

        diff = np.sum((gt_vertices_areas - barycenters_areas))
        assert pytest.approx(diff, 0, 1e-8)
        for area in barycenters_areas:
            assert area > 0

    def test_vertex_normals(self, vertices_setup):
        mesh = Mesh(normals_unit_norm=False)
        mesh.v = self.vertices
        mesh.f = self.faces
        normals = mesh.normals

        faces_areas = np.expand_dims(mesh.areas, 1)
        weighted_areas = faces_areas * normals
        vertex_face_adj = mesh.vertex_face_adjacency().todense().astype(np.int).tolist()
        vertex_face_adj = [np.where(np.array(row) == 1)[0] for row in vertex_face_adj]

        gt_vertices_normals = []
        gt_normalized_vertices_normals = []
        for v in vertex_face_adj:
            normal = np.sum(weighted_areas[v, :], 0)
            gt_vertices_normals.append(normal)
            weighted_normal = normal / np.linalg.norm(normal)
            gt_normalized_vertices_normals.append(weighted_normal)

        gt_vertices_normals = np.array(gt_vertices_normals)
        gt_normalized_vertices_normals = np.array(gt_normalized_vertices_normals)

        vertices_normals = mesh.vertex_normals
        mesh = Mesh(normals_unit_norm=True)
        mesh.v = self.vertices
        mesh.f = self.faces
        normalized_vertices_normals = mesh.vertex_normals

        diff = np.sum(np.abs(vertices_normals - gt_vertices_normals))
        norm_diff = np.sum(np.abs(normalized_vertices_normals -
                                  gt_normalized_vertices_normals))

        assert pytest.approx(diff, 0, 1e-8)
        assert pytest.approx(norm_diff, 0, 1e-8)

    def test_euler_characteristic(self, vertices_setup):
        mesh = Mesh(normals_unit_norm=False)
        mesh.v = self.vertices
        mesh.f = self.faces
        ec = mesh.euler_characteristic

        # We have a cube where each face is represented as 3 triangles, so overall
        # F = 18, V = 13, E = 33
        # Note that we have 4 triangles with 'redundant' connection, i.e.
        # vertices: 0-2, 0 - 6, 4 - 6, 9 - 11, which can be avoided
        # by using 4-vertices faces for those triangles.
        # If we would remove those connections we would get E = 29 and the
        # Euler characteristic would have been 2, but since we assume that each face
        # is defined from exactly 3 vertices in the mesh, this is currently unavoidable,
        # and leads to E = 33.
        assert ec == -2

        # Test the the Euler invariance theorem holds for a valid sample shape
        data_dir = os.path.join(PROJECT_ROOT, 'data', 'off_files')
        file = glob.glob(os.path.join(data_dir, '*.off'))[0]

        # Load Mesh
        mesh = Mesh(file)
        ec = mesh.euler_characteristic

        assert ec == 2

    def test_gaussian_curvature(self, vertices_setup):
        mesh = Mesh(normals_unit_norm=False)
        mesh.v = self.vertices
        mesh.f = self.faces

        gaussian_curvatures = mesh.gaussian_curvature
        for curve in gaussian_curvatures:
            assert curve >= 0

        # Test that the Gauss Bonnet Theorem holds for the an exemplary shape we
        # were given
        data_dir = os.path.join(PROJECT_ROOT, 'data', 'off_files')
        file = glob.glob(os.path.join(data_dir, '*.off'))[0]

        # Load Mesh
        mesh = Mesh(file)

        vertices_areas = mesh.barycenters_areas
        gaussian_curvatures = mesh.gaussian_curvature
        gauss_bonnet_scalar = gaussian_curvatures @ vertices_areas
        diff = np.abs(gauss_bonnet_scalar - (2 * np.pi * mesh.euler_characteristic))
        assert pytest.approx(diff, 0, 1e-8)

    def test_vertices_centroid(self, simple_cube):
        mesh = Mesh(normals_unit_norm=False)
        mesh.v = self.vertices
        mesh.f = self.faces

        centroid = mesh.vertices_centroid
        diff = np.sum(np.abs(centroid == np.array([0.5, 0.5, 0.5])))

        assert pytest.approx(diff, 0, 1e-8)

    def test_distances_from_centroid(self, simple_cube):
        mesh = Mesh(normals_unit_norm=False)
        mesh.v = self.vertices
        mesh.f = self.faces

        distances = mesh.distance_from_centroid()

        assert distances.shape == (len(mesh.v), )

        gt_distances = np.array(
            [(3 * (0.5 ** 2)) ** 0.5, ] * len(self.vertices)
        )
        diff = distances - gt_distances

        assert pytest.approx(diff, 0, 1e-8)



