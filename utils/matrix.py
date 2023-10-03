"""
This module provides functions for working with matrices.

The module relies on the NumPy library to represent matrices as ndarrays.

"""
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


def add_small_to_big_matrix_2d_periodically(
    big_matrix, small_matrix, add_x, add_y, added_flag=True
):
    """This function adds a small matrix to a big matrix periodically.
    The small matrix is added to the big matrix at the position (add_x, add_y).
    The added_flag is used to determine whether the small matrix is added to \
    the big matrix. If the added_flag is True, the small matrix is added to \
    the big matrix.
    """
    rows_small, cols_small = small_matrix.shape
    rows_big, cols_big = big_matrix.shape

    if rows_small > rows_big or cols_small > cols_big:
        raise ValueError("The small matrix is exactly bigger")

    length = cols_small // 2
    height = rows_small // 2
    x_added_indices = []
    y_added_indices = []

    start_x = add_x - length
    end_x = start_x + cols_small - 1
    x_range = list(range(start_x, end_x + 1))
    start_y = add_y - height
    end_y = start_y + rows_small - 1
    y_range = list(range(start_y, end_y + 1))

    overflow_left = add_x - length < 0
    overflow_right = add_x + length > cols_big - 1
    overflow_top = add_y - height < 0
    overflow_bottom = add_y + height > rows_big - 1
    overflow_x = overflow_left or overflow_right
    overflow_y = overflow_top or overflow_bottom

    if not (overflow_x or overflow_y):
        if added_flag:
            big_matrix[np.ix_(y_range, x_range)] += small_matrix
        x_added_indices.extend(x_range)
        y_added_indices.extend(y_range)
        return big_matrix, x_added_indices, y_added_indices

    # If it is out of boundary, separate the range to stepwise
    x_stepwise = []
    y_stepwise = []

    if overflow_left:
        x_right = list(range(0, end_x + 1))
        x_right_length = len(x_right)
        x_left_length = cols_small - x_right_length
        x_left_start = start_x + cols_big
        x_left_end = x_left_start + x_left_length - 1
        x_left = list(range(x_left_start, x_left_end + 1))
        x_left_small = list(range(0, x_left_length))
        x_right_small = list(range(x_left_length, cols_small))
        x_stepwise = [x_left, x_left_small, x_right, x_right_small]

    if overflow_right:
        x_left = list(range(start_x, cols_big))
        x_left_length = len(x_left)
        x_right_length = cols_small - x_left_length
        x_right_start = 0
        x_right_end = x_right_start + x_right_length
        x_right = list(range(x_right_start, x_right_end))
        x_left_small = list(range(0, x_left_length))
        x_right_small = list(range(x_left_length, cols_small))
        x_stepwise = [x_left, x_left_small, x_right, x_right_small]

    if overflow_top:
        y_bottom = list(range(0, end_y + 1))
        y_bottom_length = len(y_bottom)
        y_top_length = rows_small - y_bottom_length
        y_top_start = start_y + rows_big
        y_top_end = y_top_start + y_top_length
        y_top = list(range(y_top_start, y_top_end))
        y_top_small = list(range(0, y_top_length))
        y_bottom_small = list(range(y_top_length, rows_small))
        y_stepwise = [y_top, y_top_small, y_bottom, y_bottom_small]

    if overflow_bottom:
        y_top = list(range(start_y, rows_big))
        y_top_length = len(y_top)
        y_bottom_length = rows_small - y_top_length
        y_bottom_start = 0
        y_bottom_end = y_bottom_start + y_bottom_length - 1
        y_bottom = list(range(y_bottom_start, y_bottom_end + 1))
        y_bottom_small = list(range(0, y_bottom_length))
        y_top_small = list(range(y_bottom_length, rows_small))
        y_stepwise = [y_top, y_top_small, y_bottom, y_bottom_small]

    # Use the stepwise range to assign the value
    if overflow_x and not overflow_y:
        if added_flag:
            big_matrix[np.ix_(y_range, x_stepwise[0])] += small_matrix[
                :, x_stepwise[1]
            ]
            big_matrix[np.ix_(y_range, x_stepwise[2])] += small_matrix[
                :, x_stepwise[3]
            ]
        x_added_indices.extend(x_stepwise[0])
        x_added_indices.extend(x_stepwise[2])
        y_added_indices.extend(y_range)

    if not overflow_x and overflow_y:
        if added_flag:
            big_matrix[np.ix_(y_stepwise[0], x_range)] += small_matrix[
                y_stepwise[1], :
            ]
            big_matrix[np.ix_(y_stepwise[2], x_range)] += small_matrix[
                y_stepwise[3], :
            ]
        x_added_indices.extend(x_range)
        y_added_indices.extend(y_stepwise[0])
        y_added_indices.extend(y_stepwise[2])

    if overflow_x and overflow_y:
        if added_flag:
            big_matrix[np.ix_(y_stepwise[0], x_stepwise[0])] += small_matrix[
                np.ix_(y_stepwise[1], x_stepwise[1])
            ]
            big_matrix[np.ix_(y_stepwise[2], x_stepwise[0])] += small_matrix[
                np.ix_(y_stepwise[3], x_stepwise[1])
            ]
            big_matrix[np.ix_(y_stepwise[0], x_stepwise[2])] += small_matrix[
                np.ix_(y_stepwise[1], x_stepwise[3])
            ]
            big_matrix[np.ix_(y_stepwise[2], x_stepwise[2])] += small_matrix[
                np.ix_(y_stepwise[3], x_stepwise[3])
            ]
        x_added_indices.extend(x_stepwise[0])
        x_added_indices.extend(x_stepwise[2])
        y_added_indices.extend(y_stepwise[0])
        y_added_indices.extend(y_stepwise[2])

    return big_matrix, x_added_indices, y_added_indices


if __name__ == "__main__":
    pass
