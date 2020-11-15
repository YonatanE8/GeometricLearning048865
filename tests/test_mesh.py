from src.HW1.utils.io import read_off
from src.HW1.utils.mesh import Mesh
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
            [3, 6, 4],
            [8, 1, 9],
            [0, 1, 8],
            [1, 2, 9],
            [2, 3, 9],
            [9, 3, 11],
            [3, 4, 11],
            [4, 11, 5],
            [11, 5, 12],
            [5, 12, 6],
            [6, 7, 12],
            [7, 8, 12],
            [0, 8, 7],
            [8, 9, 10],
            [8, 10, 12],
            [10, 11, 12],
        ]

    def test_mesh_init(self):
        data_dir = os.path.join(PROJECT_ROOT, 'data', 'example_off_files')
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

        cols = np.array([0, 1, 4, 14, 3, 4, 5, 1, 5, 6, 0, 1, 2, 6, 7, 8, 2,
                         8, 9, 9, 10, 11, 0, 2, 11, 12, 12, 13, 14, 3, 4, 13, 14, 15,
                         16, 3, 5, 6, 7, 15, 15, 16, 17, 7, 8, 9, 10, 17, 10, 11, 12,
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

        Avv_gt = np.array([[False, False, False, True, False, False, False, False, True,
                            False, False, False, False],
                           [False, False, False, False, False, False, False, False,
                            True,
                            True, False, False, False],
                           [False, False, False, True, False, False, False, False,
                            False,
                            True, False, False, False],
                           [True, False, True, False, True, False, True, False, False,
                            True, False, True, False],
                           [False, False, False, True, False, False, False, False,
                            False,
                            False, False, True, False],
                           [False, False, False, False, False, False, False, False,
                            False,
                            False, False, True, True],
                           [False, False, False, True, False, False, False, False,
                            False,
                            False, False, False, True],
                           [False, False, False, False, False, False, False, False,
                            True,
                            False, False, False, True],
                           [True, True, False, False, False, False, False, True, False,
                            True, True, False, True],
                           [False, True, True, True, False, False, False, False, True,
                            False, False, False, False],
                           [False, False, False, False, False, False, False, False,
                            True,
                            False, False, False, True],
                           [False, False, False, True, True, True, False, False, False,
                            False, False, False, True],
                           [False, False, False, False, False, True, True, True, True,
                            False, True, True, False]])

        assert np.sum(1 - (np.array(Avv.todense()) ==
                           Avv_gt).reshape((-1,)).astype(np.int)) == 0

        assert np.sum(np.diag(np.array(Avv.todense()).astype(np.int))) == 0

    def test_vertices_degree(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces

        Vd = mesh.vertex_degree()
        Vd_gt = np.array(
            [2, 2, 2, 6, 2, 2, 2, 2, 6, 4, 2, 4, 6]
        )

        assert Vd.shape == (len(self.vertices),)
        assert np.sum(Vd - Vd_gt) == 0

    def test_faces_normals(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces

        normals = mesh.faces_normals(unit_norm=False)
        normed_normals = mesh.faces_normals(unit_norm=True)
        norms = np.sqrt(np.sum(np.power(normed_normals, 2), 1)).tolist()

        assert normals.shape == (len(mesh.f), 3)
        assert normed_normals.shape == (len(mesh.f), 3)
        for norm in norms:
            pytest.approx(norm, 1.0, 1e-3)

        # directions = np.array([
        #     [0, 0, -1],
        #     [0, -1, 0],
        #     [1, 0, 0],
        #     [0, 1, 0],
        #     [-1, 0, 0],
        #     [0, 0, 1],
        # ])
        #
        # normals = normals.astype(np.int)
        # normed_normals = normed_normals.astype(np.int)
        #
        # assert np.sum(directions - normals) == 0
        # assert np.sum(directions - normed_normals) == 0

    def test_barycenters(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces

    def test_areas(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces

    def test_barycenters_areas(self, vertices_setup):
        mesh = Mesh()
        mesh.v = self.vertices
        mesh.f = self.faces
