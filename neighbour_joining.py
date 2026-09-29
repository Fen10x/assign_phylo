import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes

class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence

def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """

    def calculate_q_matrix(distances):
        #calculate the row sums of the distance matrix
        r_sums = distances.sum(axis=1)
       
        #initialize Q-matrix with 0s
        q = np.zeros_like(distances)    
       
        n = len(distances)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                q[i, j] = (
                    (n - 2) * distances[i, j]
                    - r_sums[i]
                    - r_sums[j]
                )

        return q

    working_distances = distances.copy()
    working_labels = labels.copy()

    while len(working_labels) > 2:
        #1. Based on the current distance matrix, calculate the Q-matrix
        q_matrix = calculate_q_matrix(working_distances)

        #2. Find the pair of nodes with the minimum Q value
        i, j = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)

        #3. Make a new node that joins the nodes i and j, and connect the new node to the central node. 

        # Calculate the distance from the new node to each of the joined nodes

        # Calculate the distances to the new node

        # Create a new label for the joined node

        # Update the distance matrix and labels

    q_matrix = calculate_q_matrix(working_distances)

    print("Q-matrix:\n", q_matrix)

############################################ break ############################################

distances = np.array(
    [
        [0, 14, 14, 12],
        [0, 0, 16, 14],
        [0, 0, 0, 6],
        [0, 0, 0, 0],
    ],
    dtype=float,
)
distances = distances + distances.T

labels = list("ABCD")
neighbor_joining(distances, labels)