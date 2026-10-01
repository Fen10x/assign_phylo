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

    # create copies of the distances and labels to avoid modifying the original inputs
    working_distances = distances.copy()
    working_labels = labels.copy()

    # initialize the leaf nodes
    working_nodes = {}
    for label in working_labels:
        working_nodes[label] = Node(label, None, 0, None, 0)

    while len(working_labels) > 2:
        #1. Based on the current distance matrix, calculate the Q-matrix
        q_matrix = calculate_q_matrix(working_distances)

        #2. Find the pair of nodes with the minimum Q value
        i, j = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)

        #get the labels of the nodes to be joined
        label_i = working_labels[i]
        label_j = working_labels[j]

        new_label = label_i + label_j

        #3. Calculate the distance from the new node to each of the joined nodes
        i_distance = 1/2 * working_distances[i, j] + (1/(2*(len(working_labels)-2))) * (working_distances[i, :].sum() - working_distances[j, :].sum())
        j_distance = working_distances[i, j] - i_distance

        #4. Make a new node that joins the nodes i and j, and connect the new node to the central node.
        node_joined = Node(new_label, 
                           working_nodes[label_i], 
                           i_distance, 
                           working_nodes[label_j], 
                           j_distance)
        
        working_nodes[new_label] = node_joined
        
        # Calculate the distances to the new node
        new_distances = []

        for k in range(len(working_labels)):
            if k == i or k == j:
                continue

            distance = (
                working_distances[i, k]
                + working_distances[j, k]
                - working_distances[i, j]
            ) / 2

            new_distances.append(distance)

        # Update the distance matrix and labels
        new_matrix = np.zeros(
            (len(working_labels) - 1, len(working_labels) - 1)
        )
        #identify the remaining indices after removing i and j
        remaining = [
            k for k in range(len(working_labels))
            if k != i and k != j
        ]
        #fill in the new distance matrix with the distances to the new node
        for new_i, old_i in enumerate(remaining):
            for new_j, old_j in enumerate(remaining):
                new_matrix[new_i, new_j] = working_distances[old_i, old_j]
        #add the new node to the end of the distance matrix
        for new_k, distance in enumerate(new_distances):
            new_matrix[new_k, -1] = distance
            new_matrix[-1, new_k] = distance

        #update the working distances and labels
        working_distances = new_matrix
        remaining_labels = [
            working_labels[k]
            for k in remaining
        ]
        remaining_labels.append(new_label)
        working_labels = remaining_labels

        #join final two nodes and return the root
    
    #join final two nodes and return the root
    label_i = working_labels[0]
    label_j = working_labels[1]

    root_distance = working_distances[0, 1] / 2

    root = Node(
        "ROOT",
        working_nodes[label_i],
        root_distance,
        working_nodes[label_j],
        root_distance
    )

    return root

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