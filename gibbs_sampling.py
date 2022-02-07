import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap


class LinearColormap(LinearSegmentedColormap):
    """
    This class makes it easier to create custom colormaps
    """
    def __init__(self, name, segmented_data, index=None, **kwargs):
        if index is None:
            index = np.linspace(0, 1, len(segmented_data['red']))
            for key in segmented_data:
                segmented_data[key] = zip(index, segmented_data[key])
        segmented_data = dict((key, [(x, y, y) for x, y in segmented_data[key]]) for key in segmented_data)
        LinearSegmentedColormap.__init__(self, name, segmented_data, **kwargs)


alph_spec = {
    'red': [.0, .0],
    'green': [0., 0.],
    'blue': [0., .0],
    'alpha': [1., .0]
}
# create a colormap that starts at black and gradually lowers the opacity to 0
alph_cmap = LinearColormap('alph_cmap', alph_spec)


def kde(pts: np.ndarray, xx: np.ndarray, yy: np.ndarray, sigma: float=.0005):
    """
    Kernel density estimation used for visualizing estimated distributions
    :param pts: points to use for the density estimation
    :param xx: meshgrid for the x coordinates; a numpy array of shape [N, M]
    :param yy: meshgrid for the y coordinates; a numpy array of shape [N, M]
    :param sigma: the variance of each Gaussian used for the KDE
    :return: the probability map described by the points; a numpy array of shape [N, M]
    """
    d = np.concatenate([xx.flatten()[:, None], yy.flatten()[:, None]], axis=1)
    p = logsumexp(-0.5 * np.sum((d[:, None] - pts[None, :]) ** 2, axis=-1) / sigma, axis=1)
    p = np.exp(p - logsumexp(p))
    return p.reshape(xx.shape)


def gibbs(p: np.ndarray, T: int, N: int, strt: np.ndarray=None):
    """
    Gibbs sampling algorithm with a discrete probability map
    :param p: the probability map to use, a numpy array of shape [N, M]
    :param T: number of time steps to sample
    :param N: number of particles to sample
    :param strt: the starting positions of the particles on the grid described by p
    :return: a numpy array of shape [T, N, 2] of the sampling steps using the Gibbs sampling algorithm
    """
    if strt is None: strt = np.array([np.random.choice(p.shape[0], 1)[0], np.random.choice(p.shape[1], 1)[0]])
    points = np.zeros((T, N, 2))
    points[0, :, :] = strt[None, :]
    for i in tqdm(range(1, T)):
        for j in range(N):
            # sample the x coordinate every even iteration
            if not i%2:
                points[i, j, 0] = points[i-1, j, 0]
                prob = p[int(points[i-1, j, 0]), :]
                prob = prob/np.sum(prob)
                points[i, j, 1] = np.random.choice(p.shape[1], 1, p=prob)[0]
            # sample the y coordinate every odd iteration
            else:
                points[i, j, 1] = points[i-1, j, 1]
                prob = p[:, int(points[i-1, j, 1])]
                prob = prob/np.sum(prob)
                points[i, j, 0] = np.random.choice(p.shape[0], 1, p=prob)[0]
    return points


def create_prob_map(n_centers, x_var: float=.01, y_var: float=.01, corr: float=.5, separation: float=0, width: float=.8,
                    range: tuple=tuple([0, 1]), resolution: int=100, return_centers: bool=False):
    """
    Creates a probability prob_map with the specified number of peaks and separation between them
    :param n_centers: the number of peaks the distribution should have
    :param x_var: the variance of each cluster in the x direction
    :param y_var: the variance of each cluster in the y direction
    :param corr: the correlation between the x and y directions in each of the clusters
    :param separation: the separation between the two main modes of the distribution as a float in the range [0, 1]
    :param width: dictates how far from the main modes of the distribution the smaller modes can be; must be more than 0
    :param range: the range of the distribution - you can just keep it at [0, 1] as it is right now
    :param resolution: the number of cells the distribution should have along each axis
    :param return_centers: a flag indicating whether the centers should be returned or not
    :return: the distribution prob_map as well as the meshgrid ranges
    """
    # set up grid ranges and mesh
    drange = range[1]-range[0]
    mrange = np.mean(range)
    x, y = np.linspace(range[0], range[1], resolution), np.linspace(range[0], range[1], resolution)
    xx, yy = np.meshgrid(x, y)
    d = np.concatenate([xx.flatten()[:, None], yy.flatten()[:, None]], axis=1)

    # set up probability parameters
    width = np.array([width, width])[None, :]
    clusts = np.array([
        [mrange + 0.5*drange*separation, mrange + 0.5*drange*separation],
        [mrange - 0.5*drange*separation, mrange - 0.5*drange*separation],
    ])

    # sample centers
    centers = np.clip(clusts[np.random.choice(2, n_centers), :] + width*(np.random.rand(n_centers, 2) - .5), .1, .9)

    # create covariances
    cov_xy = corr * np.sqrt(y_var) * np.sqrt(x_var)
    cov = np.array([[x_var, cov_xy], [cov_xy, y_var]])
    prec = np.linalg.inv(cov)

    # create distribution
    p = logsumexp(
        -0.5 * np.sum((d[:, None] - centers[None, :]) * ((d[:, None] - centers[None, :]) @ prec[None, ...]), axis=-1),
        axis=1)

    p = np.exp(p - logsumexp(p))
    p = p.reshape(xx.shape)
    if not return_centers: return p, x, y
    else: return (p, x, y), (centers, cov, prec)


def single_particle_animation(pts, prob_map, x, y, ell: int=5, fps: int=10, save_p: str= 'gibbs.gif'):
    """
    Make animation that shows dynamics of a single particle over time
    :param pts: a list/numpy array of shape [T, N, 2] of N different particles for T different time steps (this if
                the format of the output from the gibbs(...) function)
    :param prob_map: the 2D map of the density that was sampled from
    :param x: the range of x over the 2D map (essentially np.linspace(...))
    :param y: the range of y over the 2D map (essentially np.linspace(...))
    :param ell: to help give a sense of the trajectory of the particles, after-images of the previous steps are
                retained (gradually fading); this parameter controls HOW many previous steps can be seen
    :param fps: the frames per second of the animation
    :param save_p: where the animation should be saved (extension should be included)
    """
    global text
    xx, yy = np.meshgrid(x, y)

    # create initial figure
    fig, ax1 = plt.subplots()
    ax1.contourf(xx, yy, prob_map, 15, cmap='copper', alpha=.75)
    ax1.axis('off')
    scat = ax1.scatter([-2], [-2], 30, 'k', alpha=.5)
    norm = plt.Normalize(0, 1)
    scat = ax1.scatter([-2], [-2], 30, c=[.1], cmap=alph_cmap, norm=norm)
    text = ax1.text(.01, .01, f'0/{pts.shape[0]}', color='r', verticalalignment='bottom', horizontalalignment='left')

    def init():
        ax1.set_xlim(np.min(x), np.max(x))
        ax1.set_ylim(np.min(y), np.max(y))
        ax1.set_aspect(1)
        return scat,

    pbar = tqdm(range(pts.shape[0]))

    def update(frame):
        """
        Plots the relevant information each frame
        """
        global text
        pbar.update(1)  # update the progress-bar
        text.set_text(f'{frame + 1}/{pts.shape[0]}')  # add text for iteration number in animation
        if frame > 0:  # if the frame is larger than 0 (not the first), show past iterations as well
            data = np.concatenate([pts[np.max([frame-ell, 0]):frame+1, 0:1, 1].squeeze()[:, None],
                                   pts[np.max([frame-ell, 0]):frame+1, 0:1, 0].squeeze()[:, None]], axis=1)
        else: data = np.concatenate([pts[0, 0:1, 1][:, None],
                                     pts[0, 0:1, 0][:, None]], axis=1)
        scat.set_offsets(data)
        scat.set_array(np.linspace(1, 0, len(data)))
        return scat,

    ani = FuncAnimation(fig, update, frames=pts.shape[0], init_func=init, blit=True, )
    ani.save(save_p, fps=fps)


def multi_particle_animation(pts, prob_map, x, y, show_factor: int=1, m: int=5, k_sig: float=0.001, n_pts: int=50,
                             seconds: int=7, save_p: str='gibbs.gif'):
    """
    Make animation that shows the dynamics of multiple particles at once
    :param pts: a list/numpy array of shape [T, N, 2] of N different particles for T different time steps (this if
                the format of the output from the gibbs(...) function)
    :param prob_map: the 2D map of the density that was sampled from
    :param x: the range of x over the 2D map (essentially np.linspace(...))
    :param y: the range of y over the 2D map (essentially np.linspace(...))
    :param show_factor: the skips between frames that should be shown - if show_factor is 2, every second frame will
                        be shown, if it's 5 every 5-th frame etc.
    :param m: to help stabilize the KDE approximation of the distribution that is sampled, the particles from the m
              previous frames are also used
    :param k_sig: the standard deviation used for the KDE of the sampled distribution
    :param n_pts: the number of points to actually show in each from of the animation - if there are too many, it
                  becomes hard to differentiate between them
    :param seconds: number of seconds the animation should take
    :param save_p: where the animation should be saved (extension should be included)
    """
    global cont, c

    xx, yy = np.meshgrid(x, y)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # create contour for true distribution
    ax1.contourf(xx, yy, prob_map, 15, cmap='copper', alpha=.75)
    ax1.axis('off')

    # scatter of points over true distribution
    scat = ax1.scatter([-2], [-2], 30, 'k', alpha=.5)

    # text to show iteration number of the algorithm
    text = ax1.text(.01, .01, f'0/{pts.shape[0]}', color='r', verticalalignment='bottom', horizontalalignment='left')

    # create another contour to show the distribution approximated by the algorithm
    cont = ax2.contourf(xx, yy, kde(np.concatenate([pts[0, :, 0][:, None], pts[0, :, 1][:, None]], axis=1),
                                    xx, yy), 15, cmap='copper', alpha=.75)
    ax2.axis('off')

    def init():
        ax1.set_xlim(np.min(x), np.max(x))
        ax1.set_ylim(np.min(y), np.max(y))
        ax1.set_aspect(1)

        ax2.set_xlim(np.min(x), np.max(x))
        ax2.set_ylim(np.min(y), np.max(y))
        ax2.set_aspect(1)
        return scat,

    c = 0
    maps = np.zeros([m, *prob_map.shape])
    pbar = tqdm(range(pts.shape[0] // show_factor))

    def update(frame):
        """
        Plots the relevant information each frame
        """
        global cont, c
        pbar.update(1)

        frame = show_factor * frame  # show only part of the sampling process (only every show_factor frames are shown)
        text.set_text(f'{frame + 1}/{pts.shape[0]}')  # update plot text to show correct iteration

        # define the data for the scatter plot, showing only n_pts points
        data = np.concatenate([pts[frame, :n_pts, 0][:, None], pts[frame, :n_pts, 1][:, None]], axis=1)
        scat.set_offsets(data)

        # collect points in order to calculate the distribution approximated in each frame, using KDE
        data = np.concatenate([pts[frame, :, 0][:, None], pts[frame, :, 1][:, None]], axis=1)
        for col in cont.collections: plt.gca().collections.remove(col)
        if frame > m:  # if enough frames passed, take the mean over the last m KDEs found
            map_tmp = kde(data, xx, yy, sigma=k_sig)
            maps[c % m] = map_tmp
            c += 1
            map = np.mean(maps, axis=0)
        else:
            map = kde(data, xx, yy, sigma=k_sig)
        cont = ax2.contourf(xx, yy, map, 15, cmap='copper', alpha=.75)  # plot contours of approx. distribution

        return scat,

    # create and save animation
    ani = FuncAnimation(fig, update, frames=pts.shape[0]//show_factor, init_func=init, blit=True, )
    ani.save(save_p, fps=(pts.shape[0]//show_factor)//seconds)


if __name__ == '__main__':
    p, x, y = create_prob_map(n_centers=20, separation=0, width=1)
    pts = gibbs(p, T=200, N=1, strt=np.array([p.shape[0]-1, 0])).astype(int)
    pts[:, :, 0] = x[pts[:, :, 0]]
    pts[:, :, 1] = y[pts[:, :, 0]]
    xx, yy = np.meshgrid(x, y)
    plt.figure()
    plt.contourf(xx, yy, p, 15, cmap='copper', alpha=.75)
    plt.axis('off')
    plt.show()

    # multi_particle_animation(pts, p, x, y, k_sig=0.001)
    single_particle_animation(pts, p, x, y, save_p='gibbs_single.gif')
