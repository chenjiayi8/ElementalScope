"""
This module provides functions for working with images.
"""

import numpy as np


def imshowpair(A, B, method="diff"):
    """
    Display a pair of images side-by-side or overlaid.

    Parameters:
    - A, B: Input images
    - method: Visualization method. Options are:
        'diff': Display the absolute difference between A and B
        'blend': Blend A and B using 50% of each
        'checkerboard': Checkerboard pattern of A and B
        'falsecolor': Overlay A and B with different colors
    """
    if A.shape != B.shape:
        raise ValueError("Both images must have the same dimensions.")

    if method == "diff":
        output = np.abs(A - B)
    elif method == "blend":
        output = 0.5 * A + 0.5 * B
    elif method == "checkerboard":
        checker = np.tile(
            np.array([[1, 0], [0, 1]]), (A.shape[0] // 2, A.shape[1] // 2)
        )
        output = np.where(checker == 1, A, B)
    elif method == "falsecolor":
        output = np.stack([A, B, np.zeros_like(A)], axis=-1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return output


if __name__ == "__main__":
    pass
