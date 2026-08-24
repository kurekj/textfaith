"""Render a review as a word heat map, with optional grayed out cut words.

A direct matplotlib rendering rather than `shap.plots.text`, for two reasons
that notebook 03 established: the caches hold plain per class arrays without
base values, so rebuilding a shap Explanation would fabricate a base value, and
PNG output survives every viewer that strips HTML from notebooks or reports.

The layout is a character grid under a monospace font: each word occupies its
character count plus one, lines wrap at a fixed width, and the figure size is
derived so one character corresponds to roughly one data unit. That keeps words
from overlapping without measuring rendered text.
"""

from collections.abc import Sequence

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

MONOSPACE_CHARACTER_WIDTH_EM = 0.602
LINE_HEIGHT_EM = 1.9

ATTRIBUTION_COLORMAP = LinearSegmentedColormap.from_list(
    "attribution", ["C1", "white", "C0"]
)


def wrap_words(
    words: Sequence[str], characters_per_line: int
) -> list[tuple[int, int]]:
    """Assign each word a character column and a line, wrapping at the line width."""
    positions = []
    column = 0
    line = 0
    for word in words:
        stripped_word = word.strip()
        if column > 0 and column + len(stripped_word) > characters_per_line:
            column = 0
            line += 1
        positions.append((column, line))
        column += len(stripped_word) + 1
    return positions


def render_words(
    axis,
    words: Sequence[str],
    attribution: np.ndarray,
    cut_mask: np.ndarray | None = None,
    characters_per_line: int = 70,
    fontsize: int = 10,
) -> None:
    """Draw a review on a character grid, coloring kept words and graying cut ones.

    C0 marks words pushing toward the predicted class, C1 words pushing against
    it, intensity is magnitude relative to the strongest word in this rendering,
    so per method comparisons on the same review stay scale honest within a
    panel but are not comparable across attribution scales.
    """
    maximum_magnitude = np.abs(attribution).max()
    normalize = Normalize(-maximum_magnitude, maximum_magnitude)
    positions = wrap_words(words, characters_per_line)
    for index, (word, value) in enumerate(zip(words, attribution)):
        column, line = positions[index]
        stripped_word = word.strip()
        if cut_mask is not None and cut_mask[index]:
            axis.text(
                column, -line, stripped_word, fontsize=fontsize,
                family="monospace", ha="left", va="center", color="0.8",
            )
            continue
        background = ATTRIBUTION_COLORMAP(normalize(value))
        red, green, blue = background[:3]
        luminance = 0.299 * red + 0.587 * green + 0.114 * blue
        axis.text(
            column, -line, stripped_word, fontsize=fontsize,
            family="monospace", ha="left", va="center",
            color="white" if luminance < 0.5 else "black",
            bbox=dict(facecolor=background, edgecolor="none", boxstyle="round,pad=0.15"),
        )
    line_count = positions[-1][1] + 1
    axis.set_xlim(0, characters_per_line)
    axis.set_ylim(-line_count + 0.5, 0.5)
    axis.axis("off")


def word_figure_size(
    characters_per_line: int, line_count: int, fontsize: int
) -> tuple[float, float]:
    """Figure size in inches that gives one character roughly one data unit."""
    width = characters_per_line * fontsize * MONOSPACE_CHARACTER_WIDTH_EM / 72 * 1.08
    height = (line_count + 1) * fontsize * LINE_HEIGHT_EM / 72
    return width, height
