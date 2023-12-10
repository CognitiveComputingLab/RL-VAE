import numpy as np

from util import DynamicImporter
plt = DynamicImporter('matplotlib.pyplot')
cm = DynamicImporter('matplotlib.cm')
go = DynamicImporter('plotly.graph_objects')


def change_lightness(colors, t, scale=1):
    colors = np.asarray(colors)
    t = np.asarray(t)
    pos = t > 0.5
    neg = t <= 0.5
    t_pos = 2 * (t[pos] - 0.5) * scale
    t_neg = (1 - 2 * t[neg]) * scale
    new_colors = colors.copy()[:, 0:3]
    new_colors[pos] = (1 - t_pos)[:, None] * new_colors[pos] + t_pos[:, None] * np.ones_like(new_colors[pos])
    new_colors[neg] = (1 - t_neg)[:, None] * new_colors[neg] + t_neg[:, None] * np.zeros_like(new_colors[neg])
    return new_colors


def change_saturation(colors, t):
    colors = np.asarray(colors)
    t = np.asarray(t)
    new_colors = colors.copy()[:, 0:3]
    mean = new_colors.mean(axis=1)
    new_colors = (1 - t)[:, None] * new_colors + t[:, None] * np.ones_like(new_colors) * mean[:, None]
    return new_colors


def scatter3d(dataset=None, data=None, fig=None, layout=(), **kwargs):
    if dataset is None and data is None:
        raise ValueError("You have to provide at least one of 'dataset' or 'data'")
    if data is None:
        data = dataset.data
    kwargs = dict(mode='markers', opacity=0.4, marker=dict(size=1), line=dict(width=1)) | kwargs
    if fig is None:
        layout = dict(scene=dict(aspectmode='data')) | dict(layout)
        fig = go.Figure(layout=layout)
    if dataset is not None:
        try:
            colors = dataset.colors
        except AttributeError:
            pass
        else:
            if 'marker' in kwargs and 'color' not in kwargs['marker']:
                kwargs['marker']['color'] = colors
            if 'line' in kwargs and 'color' not in kwargs['line']:
                kwargs['line']['color'] = colors
    fig.add_trace(go.Scatter3d(**dict(x=data[:, 0], y=data[:, 1], z=data[:, 2]), **kwargs))
    return fig


def mpl_2d_plot(dataset=None, data=None, equal_aspect=True, **kwargs):
    if dataset is None and data is None:
        raise ValueError("You have to provide at least one of 'dataset' or 'data'")
    if data is None:
        data = dataset.data
    try:
        kwargs = dict(color=dataset.colors) | kwargs
    except AttributeError:
        pass
    plt.scatter(data[:, 0], data[:, 1], **kwargs)
    if equal_aspect:
        plt.gca().set_aspect('equal', 'datalim')
    plt.show()
