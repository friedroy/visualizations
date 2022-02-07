import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.special import logsumexp
from tqdm import tqdm
from gibbs_sampling import single_particle_animation, multi_particle_animation, create_prob_map
from LinearRegression import polynomial_basis_functions, spline_basis_functions, gaussian_basis_functions
from typing import Union, Callable, Tuple


def BLR_langevin(x: np.ndarray, y: np.ndarray, prior: Tuple[np.ndarray, np.ndarray, float, Callable], T: int,
                 N: int, eps: Union[float, Callable]=1e-8, save_p: str='BLR_langevin.gif', show_factor: int=1):
    """
    Create animation of posterior samples from BLR using Langevin sampling
    :param x: input points as a numpy array with length M
    :param y: output points as a numpy array with length M
    :param prior: the prior to use for sampling as the tuple (mean, precision matrix, basis_func) where basis_func is a
                  function that receives points and returns the basis functions on those points
    :param T: number of time steps to run the Langevin sampling algorithm
    :param N: number of functions to sample from the posterior
    :param eps: the step size to use - either a number or a function that returns the step size given the iteration
    :param save_p: the save path for the animation, should include the extension
    :param show_factor: the skips between frames that should be shown - if show_factor is 2, every second frame will
                        be shown, if it's 5 every 5-th frame etc.
    """
    mu, prec, noise, basis_func = prior

    H = basis_func(x)
    P = prec + H.T@H/noise
    mean = prec@mu
    Hy = H.T@y/noise

    def dx(x): return (P@x.T).T - Hy - mean

    strt = mu[None, :] + np.linalg.solve(np.linalg.cholesky(prec).T, np.random.randn(prec.shape[0], N)).T
    pts = langevin_sampling(strt, dx, T, N=N, eps=eps)

    xx = np.linspace(np.min(x) - .3*np.max(x), np.max(x) + .3*np.max(x), 200)
    Hh = basis_func(xx)
    xx = np.repeat(xx[:, None], N, axis=1)

    fig, ax = plt.subplots()

    pbar = tqdm(range(T//show_factor))

    def update(frame):
        frame = show_factor * frame
        pbar.update(1)
        ax.cla()
        ax.set_xlim(np.min(x) - .3*np.max(x), np.max(x) + .3*np.max(x))
        ax.set_ylim(np.min(y) - .2 * np.max(y), np.max(y) + .2 * np.max(y))
        text = ax.text(np.min(x) - .15, np.min(y) - .15, f'{frame+1}/{T}',
                       color='r', verticalalignment='bottom', horizontalalignment='left')
        plot = ax.plot(xx, Hh@pts[frame].T, lw=2, alpha=.6)
        ax.scatter(x, y, 20, 'k', alpha=1)
        return plot

    ani = FuncAnimation(fig, update, frames=T//show_factor, blit=True, )
    ani.save(save_p, fps=20)


def GMM_sample(gmm: tuple, T: int, N: int, eps: Union[float, Callable]=1e-8, strt: np.ndarray=None):
    """
    Use Langevin sampling in order to sample from a GMM
    :param gmm: the GMM to sample from as a tuple (mu, cov, prec) of all means, covariances and precision matrices (this
                is the same format as the output of gibbs_sampling.create_prob_mat with "return_centers" set to True)
    :param T: number of time steps to sample
    :param N: number of points to sample from the GMM
    :param eps: the step size to use - either a number or a function that returns the step size given the iteration
    :param strt: the starting positions of the particles to use
    :return: a numpy array of shape [T, N, 2] of the sampling steps using the Langevin sampling algorithm
    """
    if strt is None: strt = .1*np.random.randn(N, 2)
    mu, cov, prec = gmm
    det = np.linalg.slogdet(cov)[1]
    k = mu.shape[0]

    def dx(x):
        log_like = -0.5*np.sum(((x[:, None] - mu[None, :])@prec[None, ...])*(x[:, None] - mu[None, :]), axis=-1) \
                   - 0.5*det - np.log(2*np.pi) + np.log(1/k)
        resp = np.exp(log_like - logsumexp(log_like, axis=1)[:, None])
        return np.clip(np.sum(resp[..., None]*(x[:, None] - mu[None, :])@prec[None, ...], axis=1), -10000, 10000)
    return langevin_sampling(strt, dx, T, N, eps)


def langevin_sampling(strt, dx: Callable, T: int, N: int, eps: Union[float, Callable]=1e-8):
    """
    Unadjusted Langevin sampling algorithm
    :param strt: the starting positions of the particles
    :param dx: a function that receives multiple points and returns the gradient of the loss function at those points
    :param T: number of time steps to sample
    :param N: number of particles to sample in each time step
    :param eps: the step size to use - either a number or a function that returns the step size given the iteration
    :return: a numpy array of shape [T, N, 2] of the sampling steps using the Langevin sampling algorithm
    """
    if type(eps) == float:
        num = eps
        eps = lambda i: num

    points = np.zeros((T, N, strt.shape[1]))
    points[0, :, :] = strt
    pbar = tqdm(range(1, T))
    for i in pbar:
        pbar.set_postfix_str(f'log10(eps)={np.log10(eps(i)):.2f}')
        points[i, :, :] = points[i-1, :, :] - 0.5*eps(i)*dx(points[i-1, :, :]) + \
                          np.sqrt(eps(i))*np.random.randn(N, points.shape[-1])
    return points


(p, x, y), gmm = create_prob_map(n_centers=1, return_centers=True, corr=-.99, x_var=.03, y_var=.03, width=.1,
                                 separation=0)
xx, yy = np.meshgrid(x, y)
plt.figure()
plt.contourf(xx, yy, p, 15, cmap='copper', alpha=.75)
plt.axis('off')
plt.axis('equal')
plt.show()

T = 1000
strt, ed = .05, 1e-5
gamma = np.log(ed/strt)/np.log(1+T)
eps = lambda i: strt*((1+i)**gamma)

# pts = GMM_sample(gmm, T=2000, N=500, eps=5e-4, strt=.1*np.random.randn(500, 2)+.1)
pts = GMM_sample(gmm, T=T, N=500, eps=eps, strt=.1*np.random.randn(500, 2)+.5)
multi_particle_animation(pts, p, x, y, save_p='langevin.gif', show_factor=2, k_sig=5e-4, m=2, seconds=7)
single_particle_animation(pts[:800], p, x, y, save_p='single_langevin.gif', fps=14)


# x = np.linspace(-1, 1, 100)
#
# n_freq = 5
# amps = 1*(np.random.rand(n_freq) - 0.5)
# phases = 2*np.pi*np.random.rand(n_freq)
# freq = 10*np.random.rand(n_freq)
# f = lambda x: np.sum(np.array([amps[i]*np.cos(freq[i]*x+phases[i]) for i in range(n_freq)]), axis=0)
#
# noise = 1
# # y = 5*np.sin(3*x) + np.sqrt(noise)*np.random.randn(len(x))
# y = 10*f(x) + np.sqrt(noise)*np.random.randn(len(x))
# plt.figure()
# plt.scatter(x, y, 15, 'k', alpha=.75)
# plt.show()
#
# # d = 3
# # func = polynomial_basis_functions(d)
# # mu, prec = np.zeros(d+1), .01*np.eye(d+1)
# centers = np.arange(30)
# centers = 2*(centers/np.max(centers) - 0.5)
# func = gaussian_basis_functions(centers, .1)
# mu, prec = np.zeros(len(centers)+1), .5*np.eye(len(centers)+1)
#
# T = 500
# strt, ed = 7e-2, 1e-3
# gamma = np.log(ed/strt)/np.log(1+T)
# eps = lambda i: strt*((1+i)**gamma)
# BLR_langevin(x, y, (mu, prec, noise, func), T, N=10, eps=5e-3, show_factor=1)
# # BLR_langevin(x, y, (mu, prec, noise, func), T, N=10, eps=eps, show_factor=1)