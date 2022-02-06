import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.special import logsumexp
from matplotlib.animation import FuncAnimation


def binary_cluster(pts: np.ndarray, m: int, T: int, beta: float=1, labels: np.ndarray=None):
    N = pts.shape[0]
    if labels is None: labels = np.sign(np.random.rand(N) - .5)
    dists = np.sum(pts**2, axis=-1)[:, None] + np.sum(pts**2, axis=-1)[None, :] - 2*pts@pts.T
    Cs = np.array([np.argsort(dists[i])[1:m] for i in range(N)])
    med = np.median(dists)
    dists = np.array([1/dists[i, Cs[i]] for i in range(N)])
    dists = dists/np.sum(dists, axis=1)[:, None]

    for t in range(T):
        i = np.random.choice(N, 1)[0]
        plus_potential = np.sum(labels[Cs[i]]*beta)
        plus_potential = plus_potential - logsumexp(np.array([plus_potential, -plus_potential]), axis=0)
        if np.log(np.random.rand()) <= plus_potential: labels[i] = 1
        else: labels[i] = -1

    return labels


N = 100
T, jump = 200, 5
m, beta = 25, 1
data = np.sqrt(.5)*np.random.rand(N, 2) + .5*np.sign(np.random.rand(N)-.5)[:, None]
labels = np.sign(np.random.rand(N) - .5)

fig, ax = plt.subplots()
ax.scatter(data[:, 0][labels < 0], data[:, 1][labels < 0], 30, alpha=.6)
ax.scatter(data[:, 0][labels > 0], data[:, 1][labels > 0], 30, alpha=.6)
text = ax.text(np.min(data[:, 0]), np.min(data[:, 0]), f'0/{T*jump}', color='r',
               verticalalignment='bottom', horizontalalignment='left')
plt.axis('off')
plt.box(True)


pbar = tqdm(range(T))


def update(frame):
    global text, labels
    pbar.update(1)
    labels = binary_cluster(data, m=m, T=jump, labels=labels, beta=beta)
    ax.cla()
    ax.scatter(data[:, 0][labels < 0], data[:, 1][labels < 0], 30, alpha=.6)
    ax.scatter(data[:, 0][labels > 0], data[:, 1][labels > 0], 30, alpha=.6)
    plt.setp(ax.spines.values(), linewidth=1.5)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    x_left, x_right = plt.gca().get_xlim()
    y_low, y_high = plt.gca().get_ylim()
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)))
    ax.text(x_left, y_high, f'{jump*frame+1}/{T * jump}', color='r',
            verticalalignment='bottom', horizontalalignment='left')
    return text,


ani = FuncAnimation(fig, update, frames=T-1, blit=True, )
save_p = 'gibbs_clustering.gif'
seconds = 5
ani.save(save_p, fps=T//seconds)