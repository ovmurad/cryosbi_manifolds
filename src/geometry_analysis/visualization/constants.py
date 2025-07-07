import matplotlib.pyplot as plt

from .colormaps import truncate_colormap

COLOR_MAPS = {
    "real": truncate_colormap(plt.cm.Reds, 0.15, 1.0),
    "sim": truncate_colormap(plt.cm.Blues, 0.15, 1.0),
    "comb": truncate_colormap(plt.cm.Purples, 0.15, 1.0),
}

COLORS = {
    "real": plt.cm.Paired(5),  # Deep Red
    "sim": plt.cm.Paired(1),  # Deep Blue
}
