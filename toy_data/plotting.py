from toy_data.util import DynamicImporter
plt = DynamicImporter('matplotlib.pyplot')
cm = DynamicImporter('matplotlib.cm')
go = DynamicImporter('plotly.graph_objects')


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
