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

