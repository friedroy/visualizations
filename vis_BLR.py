import numpy as np
from typing import Callable
from bokeh.plotting import curdoc, figure
from bokeh.events import Tap, Reset, PressUp, MouseMove
from bokeh.models import PointDrawTool
from bokeh.server.server import Server
from LinearRegression import (LinearRegression, BayesianLinearRegression, polynomial_basis_functions,
                              gaussian_basis_functions, sigmoid_basis_functions)
from bokeh.models import RadioButtonGroup, Slider
from bokeh.layouts import column, row

BETA = 2
BASIS_FUNCS = [lambda centers: polynomial_basis_functions(len(centers)),
               lambda centers: gaussian_basis_functions(centers, beta=BETA),
               lambda centers: sigmoid_basis_functions(centers)]
LIMS = (-25, 25)
NPTS = 1000
XX = np.linspace(LIMS[0], LIMS[1], NPTS)
STRTX, STRTY = np.linspace(-1, 1, 2), np.linspace(-1, 1, 2)


def main(doc):
    def get_centers(x: np.ndarray, N: int) -> np.ndarray:
        if N == 1: return np.array([])
        else: return np.linspace(np.min(x), np.max(x), N-1)

    def get_pred():
        x = np.array(pts.data_source.data['x'])
        y = np.array(pts.data_source.data['y'])
        dof = dof_slider.value
        if len(x) > 0: basis_funcs = BASIS_FUNCS[basis_radio.active](get_centers(x, dof))
        else: basis_funcs = BASIS_FUNCS[basis_radio.active](get_centers(np.linspace(-10, 10, 10), dof))

        if bayes_radio.active:
            mean = np.zeros(dof)
            cov = np.eye(dof)*prior_slider.value
            mdl = BayesianLinearRegression(mean, cov, noise_slider.value, basis_funcs)
        else: mdl = LinearRegression(basis_funcs)

        if len(x) > 0: mdl = mdl.fit(x, y)
        preds = mdl.predict(XX)
        conf_int = np.zeros(len(preds)) if not bayes_radio.active else mdl.predict_std(XX)
        return preds, conf_int

    def update_plot(reset: bool=False):
        if reset:
            data = {'x': STRTX, 'y': STRTY}
            pts.data_source.data = data
        x = np.array(pts.data_source.data['x'])
        y = np.array(pts.data_source.data['y'])
        BLR = bayes_radio.active
        if BLR: prior_slider.disabled = False
        else: prior_slider.disabled = True

        # plot predictions
        if len(x) > 0 or BLR:
            pred, interval = get_pred()

            data = {'x': XX, 'y1': pred - 2 * np.sqrt(interval), 'y2': pred + 2 * np.sqrt(interval)}
            conf_int.data_source.data = data

            data = {'x': XX, 'y': pred}
            pred_line.data_source.data = data

    # create figure
    fig = figure(tools='reset', min_border=10, x_range=LIMS, y_range=LIMS,
                 toolbar_location="right", x_axis_location=None, y_axis_location=None, )
    fig.toolbar.logo = None
    fig.background_fill_color = "#fafafa"
    fig.x_range.bounds = [LIMS[0], LIMS[1]]
    fig.y_range.bounds = [LIMS[0], LIMS[1]]

    # plot data points and make tools to move them
    pts = fig.scatter(x=STRTX, y=STRTY, size=10)
    draw_tool = PointDrawTool(renderers=[pts])
    fig.add_tools(draw_tool)
    fig.toolbar.active_tap = draw_tool
    fig.on_event(Tap, lambda _: update_plot())
    fig.on_event(PressUp, lambda _: update_plot())
    fig.on_event(MouseMove, lambda _: update_plot())
    fig.on_event(Reset, lambda _: update_plot(reset=True))

    # create sliders and buttons
    noise_slider = Slider(start=.001, end=5, value=.1, step=.001, title='Sample noise',
                          sizing_mode='scale_width')
    noise_slider.on_change('value', lambda x, y, z: update_plot())

    prior_slider = Slider(start=.1, end=10, value=1, step=.1, title='Prior variance',
                          sizing_mode='scale_width')
    prior_slider.on_change('value', lambda x, y, z: update_plot())
    prior_slider.disabled = True

    dof_slider = Slider(start=1, end=10, value=2, step=1, title='Degrees of freedom',
                        sizing_mode='scale_width')
    dof_slider.on_change('value', lambda x, y, z: update_plot())

    bayes_radio = RadioButtonGroup(labels=['Regular', 'Bayesian'], active=0, sizing_mode='scale_width')
    bayes_radio.on_change('active', lambda attr, old, new: update_plot())

    basis_radio = RadioButtonGroup(labels=['Polynomial', 'Gaussian', 'Sigmoid'], active=0, sizing_mode='scale_width')
    basis_radio.on_change('active', lambda attr, old, new: update_plot())

    # plot initial predictions and variance
    x = np.array(pts.data_source.data['x'])
    y = np.array(pts.data_source.data['y'])
    pred, interval = get_pred()
    conf_int = fig.varea(x=XX, y1=pred - 2 * np.sqrt(interval), y2=pred + 2 * np.sqrt(interval), fill_alpha=.5)
    pred_line = fig.line(XX, pred, line_width=3)

    layout = row([fig,
                  column([noise_slider,
                          prior_slider,
                          dof_slider,
                          bayes_radio,
                          basis_radio])
                  ])

    doc.add_root(layout)


if __name__ == '__main__':
    server = Server({'/': main})
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
