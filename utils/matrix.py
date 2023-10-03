import numpy as np


def shuffle_2d_array(array):
    """Shuffle a 2D array."""
    array_copy = array.copy().ravel()
    np.random.shuffle(array_copy)
    array_copy = array_copy.reshape(array.shape)
    return array_copy


def is_member(matrix_a, matrix_b):
    """This function checks if the elements of matrix A are present in matrix B.
    It returns a boolean array and an index array."""
    # Flatten the matrix
    flat_a = matrix_a.ravel()
    flat_b = matrix_b.ravel()

    # Check presence
    member = np.isin(flat_a, flat_b)

    # Find location
    index_flat = np.array(
        [np.where(flat_b == a)[0][0] if a in flat_b else -1 for a in flat_a]
    )

    index_flat[~member] = -1  # Set indices of non-members to 0

    # Reshape the index array to the original shape of A
    index = index_flat.reshape(matrix_a.shape)

    return member.reshape(matrix_a.shape), index


def indexing_for_add_neighbor_mass_periodically_2d(solid):
    """
    This function finds the index for adding neighbor mass periodically.
    """
    ex = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])  # x of discrete velocities
    ey = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])  # y of discrete velocities

    rows, cols = solid.shape
    solid_in = np.zeros((rows, cols))
    solid_out = np.zeros((rows, cols))
    counter = 1  # counter for unique index
    neighbor_mass_index = np.zeros((8, rows, cols), dtype=int)

    for i in range(rows):
        for j in range(cols):
            solid_in[i, j] = counter
            counter += 1

    for i in range(1, 9):
        solid_out = np.roll(solid_in, shift=(ey[i], ex[i]), axis=(0, 1))
        _, temp_index = is_member(solid_out, solid_in)
        neighbor_mass_index[i - 1, :, :] = temp_index

    return neighbor_mass_index


def add_neighbor_mass_2d(solid, neighbor_mass_index=None):
    """
    This function uses the neighbor_mass_index to add up neighboring mass.
    """
    if neighbor_mass_index is None:
        neighbor_mass_index = indexing_for_add_neighbor_mass_periodically_2d(
            solid
        )

    # Use advanced indexing to get the neighboring mass
    neighbor_mass = np.sum(solid.ravel()[neighbor_mass_index], axis=0)

    return neighbor_mass

