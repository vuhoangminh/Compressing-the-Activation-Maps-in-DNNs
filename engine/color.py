import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from colormath.color_objects import (
    sRGBColor,
    HSVColor,
    LabColor,
    LCHuvColor,
    XYZColor,
    LCHabColor,
)
from colormath.color_conversions import convert_color


def hex_to_rgb_color(hex):
    return sRGBColor(
        *[int(hex[i + 1 : i + 3], 16) for i in (0, 2, 4)], is_upscaled=True
    )


def plot_color_palette(fig, colors, subplot, title, plt_count):
    ax = fig.add_subplot(plt_count, 1, subplot)
    for sp in ax.spines:
        ax.spines[sp].set_visible(False)
    for x, color in enumerate(colors):
        ax.add_patch(mpl.patches.Rectangle((x, 0), 0.95, 1, facecolor=color))
    ax.set_xlim((0, len(colors)))
    ax.set_ylim((0, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    plt.title(title)


def create_palette(
    start_rgb="#BBDEFB", end_rgb="#0000FF", n=16, colorspace=sRGBColor
):  # blue
    start_rgb = hex_to_rgb_color(start_rgb)
    end_rgb = hex_to_rgb_color(end_rgb)

    # convert start and end to a point in the given colorspace
    start = convert_color(start_rgb, colorspace).get_value_tuple()
    end = convert_color(end_rgb, colorspace).get_value_tuple()

    # create a set of n points along start to end
    points = list(zip(*[np.linspace(start[i], end[i], n) for i in range(3)]))

    # create a color for each point and convert back to rgb
    rgb_colors = [convert_color(colorspace(*point), sRGBColor) for point in points]

    # finally convert rgb colors back to hex
    return [color.get_rgb_hex() for color in rgb_colors]


if __name__ == "__main__":
    start_color = "#009392"
    end_color = "#d0587e"
    number_of_colors = 16
    colorspaces = (sRGBColor, HSVColor, LabColor, LCHuvColor, LCHabColor, XYZColor)

    start_rgb = hex_to_rgb_color(start_color)
    end_rgb = hex_to_rgb_color(end_color)
    fig = plt.figure(figsize=(number_of_colors, len(colorspaces)), frameon=False)

    for index, colorspace in enumerate(colorspaces):
        palette = create_palette(start_rgb, end_rgb, number_of_colors, colorspace)
        plot_color_palette(
            fig, palette, index + 1, colorspace.__name__, len(colorspaces)
        )

    plt.subplots_adjust(hspace=1.5)
    plt.show()
