from helper_functions import Node
from helper_functions import neighbor_joining
import numpy as np
import matplotlib.pyplot as plt

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
                x,
                y,
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

distances = np.array([
    [0, 5, 9, 9],
    [5, 0, 10, 10],
    [9, 10, 0, 8],
    [9, 10, 8, 0]
])

labels = ["A", "B", "C", "D"]

tree = neighbor_joining(distances, labels)

fig, ax = plt.subplots(figsize=(8, 8))
plot_nj_tree(tree, ax)

plt.show()