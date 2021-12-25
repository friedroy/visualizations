import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from tqdm import tqdm
from matplotlib.animation import FuncAnimation


def kde(pts: np.ndarray, xx: np.ndarray, yy: np.ndarray, sigma: float=.0005):
    d = np.concatenate([xx.flatten()[:, None], yy.flatten()[:, None]], axis=1)
    p = logsumexp(-0.5 * np.sum((d[:, None] - pts[None, :]) ** 2, axis=-1) / sigma, axis=1)
    p = np.exp(p - logsumexp(p))
    return p.reshape(xx.shape)


def gibbs(p: np.ndarray, T: int, N: int, strt: np.ndarray=None):
    if strt is None: strt = np.array([np.random.choice(p.shape[0], 1)[0], np.random.choice(p.shape[1], 1)[0]])
    points = np.zeros((T, N, 2))
    points[0, :, :] = strt[None, :]
    for i in tqdm(range(1, T)):
        for j in range(N):
            if not i%2:
                points[i, j, 0] = points[i-1, j, 0]
                prob = p[int(points[i-1, j, 0]), :]
                prob = prob/np.sum(prob)
                points[i, j, 1] = np.random.choice(p.shape[1], 1, p=prob)[0]
            else:
                points[i, j, 1] = points[i-1, j, 1]
                prob = p[:, int(points[i-1, j, 1])]
                prob = prob/np.sum(prob)
                points[i, j, 0] = np.random.choice(p.shape[0], 1, p=prob)[0]
    return points


x_range = [0, 1]
y_range = [0, 1]
N = 100
x, y = np.linspace(x_range[0], x_range[1], N), np.linspace(y_range[0], y_range[1], N)
xx, yy = np.meshgrid(x, y)
d = np.concatenate([xx.flatten()[:, None], yy.flatten()[:, None]], axis=1)

seperation = 0
width = np.array([.1, .1])[None, :]
clusts = np.array([
    [np.mean(x_range), np.mean(y_range) + 0.5*(y_range[1]-y_range[0])*seperation],
    [np.mean(x_range), np.mean(y_range) - 0.5*(y_range[1]-y_range[0])*seperation],
])

n_centers = 1
centers = np.clip(clusts[np.random.choice(2, n_centers), :] + width*(np.random.rand(n_centers, 2)-.5), .1, .9)

corr = 0.9
y_var = .02
x_var = .02
cov_xy = corr*np.sqrt(y_var)*np.sqrt(x_var)
cov = np.array([[x_var, cov_xy], [cov_xy, y_var]])
prec = np.linalg.inv(cov)
p = logsumexp(-0.5*np.sum((d[:, None]-centers[None, :])*((d[:, None]-centers[None, :])@prec[None, ...]), axis=-1),
              axis=1)
# p = logsumexp(-0.5*np.sum((d[:, None]-centers[None, :])**2, axis=-1)/sigma, axis=1)
p = np.exp(p - logsumexp(p))
p = p.reshape(xx.shape)

plt.figure()
plt.contourf(xx, yy, p, 15, cmap='copper', alpha=.75)
plt.axis('off')
plt.show()

pts = gibbs(p, T=500, N=1, strt=np.array([p.shape[0]-1, 0])).astype(int)
# pts = gibbs(p, T=2000, N=500, strt=np.array([p.shape[0]//2, p.shape[1]//2])).astype(int)


# fig, (ax1, ax2) = plt.subplots(1, 2)
fig, ax1 = plt.subplots()
ax1.contourf(xx, yy, p, 15, cmap='copper', alpha=.75)
ax1.axis('off')
scat = ax1.scatter([-2], [-2], 30, 'r', alpha=.75)
text = ax1.text(.01, .01, f'0/{pts.shape[0]}', color='r', verticalalignment='bottom', horizontalalignment='left')

# tmp = pts[0, :, ]
# cont = ax2.contourf(xx, yy, kde(np.concatenate([x[pts[0, :, 1]][:, None], y[pts[0, :, 0]][:, None]], axis=1),
#                                 xx, yy), 15, cmap='copper', alpha=.75)
# ax2.axis('off')


def init():
    ax1.set_xlim(x_range[0], x_range[1])
    ax1.set_ylim(y_range[0], y_range[1])
    ax1.set_aspect(1)

    # ax2.set_xlim(x_range[0], x_range[1])
    # ax2.set_ylim(y_range[0], y_range[1])
    # ax2.set_aspect(1)
    return scat,


show_factor = 1
m = 5
c = 0
maps = np.zeros([m, *p.shape])
k_sig = 0.001


def update(frame):
    print(frame)
    global cont, c, m, maps

    frame = show_factor*frame
    text.set_text(f'{frame+1}/{pts.shape[0]}')
    n_pts = 50
    # data = np.concatenate([x[pts[frame, :n_pts, 1]][:, None], y[pts[frame, :n_pts, 0]][:, None]], axis=1)
    data = np.concatenate([x[pts[frame, :, 1]][:, None], y[pts[frame, :, 0]][:, None]], axis=1)
    scat.set_offsets(data)

    # data = np.concatenate([x[pts[frame, :, 1]][:, None], y[pts[frame, :, 0]][:, None]], axis=1)
    #
    # for col in cont.collections: plt.gca().collections.remove(col)
    # if frame > 5:
    #     # data = np.concatenate([x[pts[frame-m:frame+1, :, 1]].flatten()[:, None],
    #     #                        y[pts[frame-m:frame+1, :, 0]].flatten()[:, None]], axis=1)
    #     map_tmp = kde(data, xx, yy, sigma=k_sig)
    #     maps[c % m] = map_tmp
    #     c += 1
    #     map = np.mean(maps, axis=0)
    # else:
    #     map = kde(data, xx, yy, sigma=k_sig)
    # cont = ax2.contourf(xx, yy, map, 15, cmap='copper', alpha=.75)

    return scat,


ani = FuncAnimation(fig, update, frames=pts.shape[0]//show_factor, init_func=init, blit=True,)
ani.save('gibbs.gif', fps=pts.shape[0]//7)
