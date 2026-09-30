"""Helper functions for HW3"""
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

def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")

    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    leaf_names = []
    leaf_positions = []

    def draw(node, x):

        # If this is a leaf
        if node.left is None and node.right is None:
            y = len(leaf_names)

            leaf_names.append(node.name)
            leaf_positions.append(y)

            #add node label
            ax.text(
            x,
            y,
            f"  {node.name}",
            fontsize=12,
            ha="left",
            va="center"
            )

            return y

        # Calculate the x-position of each child
        left_x = x + node.left_distance
        right_x = x + node.right_distance

        # Recursively draw the children
        left_y = draw(node.left, left_x)
        right_y = draw(node.right, right_x)

        # Draw horizontal branches
        ax.plot(
            [x, left_x],
            [left_y, left_y],
            color="black",
        )

        ax.plot(
            [x, right_x],
            [right_y, right_y],
            color="black",
        )

        # Draw vertical line connecting the two children
        ax.plot(
            [x, x],
            [left_y, right_y],
            color="black",
        )

        # Parent is halfway between the children
        y= (left_y + right_y) / 2

        # Add node label
        if node.name != "ROOT":
            ax.text(
                x-0.1,
                y+0.1,
                f"  {node.name}",
                ha="right",
                va="center"
            )

        return y

    # Start at the root
    draw(tree, 0)

    # Draw a short continuation line from the root
    root_y = (leaf_positions[0] + leaf_positions[-1]) / 2

    ax.plot(
    [-0.2, 0],
    [root_y, root_y],
    color="black"
    )

    #remove the top and right spines to match the example figure
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None



def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Parameters
    ----------
    tree: Node
        The root node of the tree.

    """
    raise NotImplementedError()


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()