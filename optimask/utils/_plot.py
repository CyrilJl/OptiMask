import matplotlib.pyplot as plt
import numpy as np


def plot(
    data,
    rows_to_keep=None,
    cols_to_keep=None,
    rows_to_remove=None,
    cols_to_remove=None,
    figsize=None,
    title=None,
    xticks=None,
    yticks=None,
    show=True,
):
    """
    Plots a 2D array with specific rows or columns highlighted or excluded using color coding.

    Args:
        data (np.ndarray): The 2D array of data to plot.
        rows_to_keep (list[int], optional): List of row indices to highlight (retain without modification).
            Rows not in this list will be shaded.
        cols_to_keep (list[int], optional): List of column indices to highlight (retain without modification).
            Columns not in this list will be shaded.
        rows_to_remove (list[int], optional): List of row indices to shade, independent of `rows_to_keep`.
        cols_to_remove (list[int], optional): List of column indices to shade, independent of `cols_to_keep`.
        figsize (tuple(float, float), optional): The size of the plot (width, height) in inches.
        title (str, optional): The title of the plot.
        xticks (list[str], optional): Labels for the x-axis ticks.
        yticks (list[str], optional): Labels for the y-axis ticks.
        show (bool, optional): Whether to display the plot immediately. Defaults to True.

    Returns:
        None

    Raises:
        ValueError: If the `data` input is not a 2D array.

    Notes:
        - Rows and columns specified in `rows_to_keep` and `cols_to_keep` remain unchanged.
        - Rows and columns in `rows_to_remove` and `cols_to_remove` will be shaded regardless of their presence in `rows_to_keep` or `cols_to_keep`.
        - If `xticks` or `yticks` are not specified, axis ticks will be hidden.

    Example:
        >>> import numpy as np
        >>> data = np.random.rand(10, 10)
        >>> plot(data, rows_to_keep=[1, 2], cols_to_remove=[3, 4], title="Sample Plot", xticks=list('ABCDEFGHIJ'), yticks=range(10))
    """
    cmap = plt.get_cmap("coolwarm")
    cmap.set_bad("grey")
    x = data.copy()
    m, n = data.shape

    if rows_to_keep is not None:
        x[[i for i in range(m) if i not in rows_to_keep]] += 1
    if cols_to_keep is not None:
        x[:, [j for j in range(n) if j not in cols_to_keep]] += 1
    if rows_to_remove is not None:
        x[rows_to_remove] += 1
    if cols_to_remove is not None:
        x[:, cols_to_remove] += 1

    if figsize:
        plt.figure(figsize=figsize)

    plt.pcolormesh(np.clip(x, 0, 1), cmap=cmap, alpha=0.9)
    plt.gca().set_aspect("equal")
    plt.title(title)

    if xticks is not None:
        plt.xticks(np.arange(n) + 0.5, xticks, rotation=90, size="x-small")
    if yticks is not None:
        plt.yticks(np.arange(m) + 0.5, yticks, size="x-small")
    if (xticks is None) and (xticks is None):
        plt.gca().axis("off")

    if show:
        plt.show()
