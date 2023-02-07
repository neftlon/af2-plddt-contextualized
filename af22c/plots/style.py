import logging
from contextlib import contextmanager
from pprint import pformat
import matplotlib.pyplot as plt
import matplotlib as mpl

def colorsampler(colormap_name="Set2", start_idx=0, warn_at_wraparound=True):
    """Return a function to generate colors from a colormap.

    `colormap_name` specifies the matplotlib colormap,
    `start_idx` specifies the initial index which colors are sampled from
    `warn_at_wraparound` if True, the function will print a warning to console if it runs out of colors"""
    idx = start_idx
    colors = mpl.colormaps[colormap_name].colors
    n = len(colors)

    def sample():
        nonlocal idx
        if idx == (n - 1) and warn_at_wraparound:
            logging.warning("Repeating colors for colormap %s, requested more colors than available (=%d)" %
                            (colormap_name, n))
        color = colors[idx % n]
        idx += 1
        idx %= n
        return color

    return sample


@contextmanager
def style(style_name="bmh", **kwargs):
    """A contextmanager to set up unified style. **kwargs are passed to the colorsampler creation."""
    sampler = colorsampler(**kwargs)
    try:
        with plt.style.context(style_name):
            yield sampler
    except IOError as err:
        raise Exception("Available styles:\n%s" % pformat(plt.style.available)) from err
