from typing import Tuple, List


def read_off(file_path: str):
    """
    A utility function for reading files in the OFF format

    :param file_path: (str) Path to the file to be loaded

    :return: ((list, list)) First list contains the list of vertices X, Y, Z
    coordinates, and the second list contains the indices of all vertices for each face.
    """

    # Open file
    with open(file_path, 'r') as file:
        # Assert that the file is in the right format
        first_line = file.readline().strip()
        assert first_line == 'OFF', "First line doesn't contain OFF, " \
                                    "not a valid OFF header"

        # Query # Vertices & # Faces
        n_vertices, n_faces, _ = [int(string)
                                  for string in file.readline().strip().split()]

        # Read all lines describing vertices
        vertices = [[
            float(vertice) for vertice in file.readline().strip().split()
        ] for _ in range(n_vertices)]

        # Read all lines describing faces
        faces = [[
            int(face) for face in file.readline().strip().split()[1:]
        ] for _ in range(n_faces)]

    return vertices, faces


def write_off(data: Tuple[List, List], save_path: str):
    """
    A utility function for writing files in the OFF format

    :param data: ((list, list)) First list contains the list of vertices X, Y, Z
    coordinates, and the second list contains the indices of all vertices for each face.
    :param save_path: (str) Path indicating where to save the file to
    (should include file name ending with .off)

    :return: None
    """

    # Assert that the given  save path is valid
    assert save_path.endswith('.off'), "Please indicate a 'save_path' that ends " \
                                       "with '.off'"

    # Unpack
    vertices, faces = data
    n_vertices = len(vertices)
    n_faces = len(faces)

    # Write the data
    with open(save_path, 'w') as file:
        # Write header
        file.write('OFF\n')

        # Write # Vertices & # Faces
        file.write(f'{n_vertices} {n_faces} 0\n')

        # Write vertices X, Y, Z coordinates
        [
            file.write(f"{vertice[0]} {vertice[1]} {vertice[2]}\n")
            for vertice in vertices
        ]

        # Write the vertices indices for each face
        [
            file.write(f"{len(face)} {' '.join([str(f) for f in face])}\n")
            for face in faces
        ]




