from src.utils.io import write_off, read_off
from src import PROJECT_ROOT

import os
import glob
import pytest
import shutil
import numpy as np


@pytest.fixture
def reference_off_files():
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'example_off_files')
    files = glob.glob(os.path.join(data_dir, '*.off'))

    return files


class TestOffIO:
    def test_read_off(self, reference_off_files):
        # Load data files
        for file in reference_off_files:
            data = read_off(file_path=file)

            # Check that the file has 2 lists, 1 for vertices & one for faces
            assert len(data) == 2
            assert (isinstance(data[0], list) or isinstance(data[0], tuple))
            assert (isinstance(data[1], list) or isinstance(data[1], tuple))

            # Check that each vertice has 3 coordinates
            for d in data[0]:
                assert len(d) == 3

    def test_write_off(self, reference_off_files):
        # Make temporary dir to write data into
        tmp_dir = os.path.join(PROJECT_ROOT, 'data', 'TMP_DIR')
        os.makedirs(tmp_dir, exist_ok=True)
        temp_file = os.path.join(tmp_dir, 'tmp_file.off')

        # Read a file, write it again and compare that the newly written file is
        # identical to the original one
        data = read_off(file_path=reference_off_files[0])
        write_off(data=data, save_path=temp_file)

        # Read the newly written file
        new_data = read_off(temp_file)

        # Compare
        vertices, faces = data
        new_vertices, new_faces = new_data

        n_vertices = len(vertices)
        n_faces = len(faces)

        assert len(new_vertices) == n_vertices
        assert len(new_faces) == n_faces

        vertices_diffs = [np.sum((np.array(vertices[v]) - np.array(new_vertices[v])))
                          for v in range(n_vertices)]
        faces_diffs = [np.sum((np.array(faces[f]) - np.array(new_faces[f])))
                       for f in range(n_faces)]

        assert np.sum(vertices_diffs) == 0.
        assert np.sum(faces_diffs) == 0

        # Remove the temporary files
        shutil.rmtree(tmp_dir)



