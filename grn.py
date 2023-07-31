import numpy as np

import random


def evaluate_grn(
    grn: np.ndarray,
    initial_expression: np.array,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-6,
) -> np.array:
    # Initialize gene expression levels
    expression = initial_expression.copy()

    # Initialize a variable to track the change in gene expression levels
    change: float = np.inf

    # Iterate until the GRN converges or the maximum number of iterations is reached
    for _ in range(max_iterations):
        if change < convergence_threshold:
            break

        # Calculate the new gene expression levels
        new_expression = grn.dot(expression)

        # Normalize the gene expression levels
        new_expression /= new_expression.sum()

        # Calculate the change in gene expression levels
        change = np.abs(new_expression - expression).max()

        # Update the gene expression levels
        expression = new_expression

    return expression

if __name__ == "__main__":
    # Initialize the GRN with a sparse matrix
    grn: np.ndarray = np.array([
        [0, 0.2, 0],
        [0.3, 0, 0],
        [0, 0, 0]
    ])

    # Initialize the gene expression levels
    initial_expression: np.array = np.array([1, 1, 1])

    # Evaluate the GRN
    final_expression: np.array = evaluate_grn(grn, initial_expression)

    print('Final gene expression levels:', final_expression)
