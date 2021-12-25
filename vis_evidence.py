import numpy as np
from typing import Callable
from bokeh.plotting import figure
from bokeh.events import Tap, PressUp, MouseMove
from bokeh.server.server import Server
from LinearRegression import BayesianLinearRegression
from bokeh.models import RadioButtonGroup, Slider, Div, PointDrawTool
from bokeh.layouts import column, row


def gaussian_basis_functions(centers: np.ndarray, beta: float) -> Callable:
    """
    Create a function that calculates Gaussian basis functions around a set of centers
    :param centers: an array of centers used by the basis functions
    :param beta: a float depicting the lengthscale of the Gaussians
    :return: a function that receives as input an array of values X of length N and returns the design matrix of the
             Gaussian basis functions, a numpy array of shape [N, len(centers)+1]
    """
    def gbf(x: np.ndarray):
        return np.concatenate([np.exp(-.5*((x-m)**2)/(beta**2))[:, None] for m in centers], axis=1)

    return gbf


BETA = 2
DIST = 5.5
FUNC = lambda x, a: 10*np.exp(-((DIST - x) ** 2) / (2 * BETA)) + a*np.exp(-((DIST + x) ** 2) / (2 * BETA))
EV_LIMS = (-10, 10)
LIMS = (-25, 25)
NPTS = 500
RES = .5
XX = np.linspace(LIMS[0], LIMS[1], NPTS)
STRTX = np.linspace(-25, 25, 100)
cov_func = lambda v1, v2, corr: np.array([[v1, corr*v1*v2], [corr*v1*v2, v2]])
amplitude = -10
noise = 1
var1, var2, corr = 1, 1, 0
eps = np.random.randn(len(STRTX))
f_x = FUNC(STRTX, amplitude) + np.sqrt(noise) * eps

means = np.array([10, -10])


def main(doc):
    global noise, var1, var2, amplitude, corr
    def update_plot():
        global noise, var1, var2, amplitude, corr
        if len(pts.data_source.data['x']) > 1:
            data = {'x': [pts.data_source.data['x'][-1]],
                    'y': [pts.data_source.data['y'][-1]]}
            pts.data_source.data = data

        if noise != noise_slider.value or var1 != var1_slider.value or var2 != var2_slider.value or \
                amplitude != amp_slider.value or corr != corr_slider.value:
            noise = noise_slider.value
            var1 = var1_slider.value
            var2 = var2_slider.value
            amplitude = amp_slider.value
            corr = corr_slider.value

            # update heatmap
            xs, ys = np.meshgrid(np.arange(EV_LIMS[0], EV_LIMS[1], RES), np.arange(EV_LIMS[0], EV_LIMS[1], RES))
            zs = np.zeros(xs.shape)
            for i in range(xs.shape[0]):
                for j in range(xs.shape[1]):
                    gbf = gaussian_basis_functions(np.array([xs[i, j], ys[i, j]]), BETA)
                    BLR = BayesianLinearRegression(theta_mean=means, theta_cov=cov_func(var1, var2, corr),
                                                   sig=noise, basis_functions=gbf)
                    zs[i, j] = BLR.log_evidence(STRTX, FUNC(STRTX, amplitude))
            zs = (zs - np.min(zs)) / (np.max(zs) - np.min(zs))
            image.data_source.data.update({'image': [zs]})

        data = {'x': STRTX, 'y': FUNC(STRTX, amplitude) + eps * np.sqrt(noise)}
        scat.data_source.data = data

        x1 = np.array(pts.data_source.data['x'])
        x2 = np.array(pts.data_source.data['y'])

        gbf = gaussian_basis_functions(np.array([x1, x2]), BETA)
        BLR = BayesianLinearRegression(theta_mean=means, theta_cov=cov_func(var1, var2, corr),
                                       sig=noise, basis_functions=gbf)
        pred = BLR.predict(XX)
        interval = BLR.predict_std(XX)

        data = {'x': XX, 'y1': pred - 2 * np.sqrt(interval), 'y2': pred + 2 * np.sqrt(interval)}
        conf_int.data_source.data = data
        data = {'x': XX, 'y': pred}
        mean_line.data_source.data = data

        BLR = BLR.fit(STRTX, FUNC(STRTX, amplitude))
        pred = BLR.predict(XX)
        data = {'x': XX, 'y': pred}
        fit_line.data_source.data = data

    # create evidence figure
    mean_fig = figure(min_border=10, x_range=EV_LIMS, y_range=EV_LIMS, tooltips=[("(x,y)", "($x, $y)")])
    mean_fig.outline_line_width = 3
    mean_fig.outline_line_color = 'black'
    mean_fig.outline_line_alpha = .7
    mean_fig.toolbar.logo = None
    mean_fig.toolbar_location = None
    mean_fig.background_fill_color = "#fafafa"

    mean_fig.x_range.bounds = EV_LIMS
    mean_fig.y_range.bounds = EV_LIMS

    mean_fig.xaxis.ticker = []
    mean_fig.yaxis.ticker = []
    mean_fig.xaxis.major_label_text_font_size = '0pt'
    mean_fig.yaxis.major_label_text_font_size = '0pt'
    mean_fig.xaxis.major_tick_line_color = None
    mean_fig.xaxis.minor_tick_line_color = None
    mean_fig.yaxis.major_tick_line_color = None
    mean_fig.yaxis.minor_tick_line_color = None
    mean_fig.xaxis.axis_label = r'mu 1'
    mean_fig.yaxis.axis_label = r'mu 2'

    # make figure with prior
    pts_fig = figure(min_border=10, x_range=LIMS, y_range=LIMS, y_axis_location='right')
    pts_fig.xgrid.grid_line_color = None
    pts_fig.ygrid.grid_line_color = None
    pts_fig.outline_line_width = 3
    pts_fig.outline_line_color = 'black'
    pts_fig.outline_line_alpha = .7
    pts_fig.toolbar.logo = None
    pts_fig.toolbar_location = None
    pts_fig.background_fill_color = "#fafafa"
    pts_fig.x_range.bounds = [LIMS[0], LIMS[1]]
    pts_fig.y_range.bounds = [LIMS[0], LIMS[1]]
    pts_fig.xaxis.major_tick_line_color = None
    pts_fig.xaxis.minor_tick_line_color = None
    pts_fig.yaxis.major_tick_line_color = None
    pts_fig.yaxis.minor_tick_line_color = None
    pts_fig.xaxis.ticker = []
    pts_fig.yaxis.ticker = []
    pts_fig.yaxis.axis_label = r'FUNC(x)'
    pts_fig.xaxis.axis_label = r'x'

    scat = pts_fig.scatter(x=STRTX, y=f_x, size=10)

    # plot evidence heatmap
    xs, ys = np.meshgrid(np.arange(EV_LIMS[0], EV_LIMS[1], RES), np.arange(EV_LIMS[0], EV_LIMS[1], RES))
    zs = np.zeros(xs.shape)
    for i in range(xs.shape[0]):
        for j in range(xs.shape[1]):
            gbf = gaussian_basis_functions(np.array([xs[i, j], ys[i, j]]), BETA)
            BLR = BayesianLinearRegression(theta_mean=means, theta_cov=cov_func(var1, var2, corr),
                                           sig=noise, basis_functions=gbf)
            zs[i, j] = BLR.log_evidence(STRTX, f_x)
    zs = (zs - np.min(zs))/(np.max(zs) - np.min(zs))
    image = mean_fig.image(image=[zs], x=EV_LIMS[0], y=EV_LIMS[0], dh=EV_LIMS[1]-EV_LIMS[0], dw=EV_LIMS[1]-EV_LIMS[0],
                           palette="Plasma256", level="image", dilate=True)

    # plot initial predictions and variance
    gbf = gaussian_basis_functions(np.array([0, 0]), BETA)
    BLR = BayesianLinearRegression(theta_mean=means, theta_cov=cov_func(var1, var2, corr),
                                   sig=noise, basis_functions=gbf)
    pred = BLR.predict(XX)
    interval = BLR.predict_std(XX)

    conf_int = pts_fig.varea(x=XX, y1=pred - 2 * np.sqrt(interval), y2=pred + 2 * np.sqrt(interval), fill_alpha=.5)
    mean_line = pts_fig.line(XX, pred, line_width=3)

    BLR = BLR.fit(STRTX, f_x)
    pred = BLR.predict(XX)
    fit_line = pts_fig.line(XX, pred, line_width=2, color='red')

    # plot data points and make tools to move them
    pts = mean_fig.scatter(x=[0], y=[0], size=10, fill_color='white', line_color='black', line_width=2)
    draw_tool = PointDrawTool(renderers=[pts])
    mean_fig.add_tools(draw_tool)
    mean_fig.toolbar.active_tap = draw_tool
    mean_fig.on_event(Tap, lambda _: update_plot())
    mean_fig.on_event(PressUp, lambda _: update_plot())
    mean_fig.on_event(MouseMove, lambda _: update_plot())

    # create sliders and buttons
    noise_slider = Slider(start=.1, end=10, value=1, step=.1, sizing_mode='scale_width', title='noise variance')
    noise_slider.on_change('value', lambda x, y, z: update_plot())

    amp_slider = Slider(start=-10, end=10, value=-10, step=1, sizing_mode='scale_width', title='mode 2 amplitude')
    amp_slider.on_change('value', lambda x, y, z: update_plot())

    var1_slider = Slider(start=.1, end=10, value=1, step=.1, sizing_mode='scale_width', title='mu 1 variance')
    var1_slider.on_change('value', lambda x, y, z: update_plot())

    var2_slider = Slider(start=.1, end=10, value=1, step=.1, sizing_mode='scale_width', title='mu 2 variance')
    var2_slider.on_change('value', lambda x, y, z: update_plot())

    corr_slider = Slider(start=-.9, end=.9, value=0, step=.1, sizing_mode='scale_width', title='correlation')
    corr_slider.on_change('value', lambda x, y, z: update_plot())

    layout = row([column([mean_fig, row([var1_slider, var2_slider]), corr_slider]),
                  column([pts_fig, noise_slider, amp_slider])])

    doc.add_root(layout)


if __name__ == '__main__':
    server = Server({'/': main})
    server.start()
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
