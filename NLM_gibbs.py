import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp
from tqdm import tqdm
from matplotlib.animation import FuncAnimation


def to_patches(im: np.ndarray, p_sz: int, m: int=None):
    if m is None:
        patches = np.zeros((im.shape[0], im.shape[1], p_sz, p_sz, 3))
        for i in range(patches.shape[0]-p_sz):
            for j in range(patches.shape[1]-p_sz):
                patches[i, j] = im[i:i + p_sz][:, j:j+p_sz]
        return patches.reshape(-1, p_sz*p_sz*3)
    xs, ys = np.random.choice(im.shape[0]-p_sz, m, replace=True), np.random.choice(im.shape[1]-p_sz, m, replace=True)
    return np.array([im[x:x+p_sz, y:y+p_sz] for (x, y) in zip(xs, ys)]).reshape(m, -1)


def gaussian_sample(noisy: np.ndarray, orig: np.ndarray, samps: np.ndarray, w: np.ndarray, noise: float) -> np.ndarray:
    lat = samps.shape[0]//2
    mu = np.sum(w[:, None]*samps, axis=0)/np.sum(w)

    est_var = np.mean((orig - mu)**2)

    cov = ((w[:, None]*(samps-mu)).T @ (samps-mu))/np.sum(w)
    _, s, v = np.linalg.svd(cov, full_matrices=True)
    W = v[:lat, :].T * np.sqrt(np.clip(s[:lat] - est_var, 0, 100))[None, :]
    # p = np.mean(np.clip(s[lat:] - est_var, 0, 100)) + 1e-6
    p = 1e-6
    M_inv = np.linalg.inv(W@W.T + np.eye(W.shape[0])*p)

    mean = np.linalg.solve(M_inv + np.eye(M_inv.shape[0])/noise, M_inv@mu + noisy/noise)
    chol = np.linalg.cholesky(M_inv + np.eye(M_inv.shape[0])/noise).T

    return mean + np.linalg.solve(chol, np.random.randn(chol.shape[1]))


def gibbs_denoise(im: np.ndarray, init: np.ndarray, noise: float, p_sz: int, its: int, k: int=10,
                  beta: float=.1, m: int=None) -> np.ndarray:
    outp = init.copy()
    for i in range(its):
        patches = to_patches(outp, p_sz, m)
        x = np.random.choice(init.shape[0] - p_sz, 1, replace=True)[0]
        y = np.random.choice(init.shape[1] - p_sz, 1, replace=True)[0]

        patch = outp[x:x+p_sz, y:y+p_sz].flatten()
        noisy = im[x:x+p_sz, y:y+p_sz].flatten()
        dists = np.sum((patch[None, ...] - patches)**2, axis=1)
        inds = np.argsort(dists)[1:k+1]

        w = -0.5*beta*dists[inds]
        w = np.exp(w - logsumexp(w, axis=0))
        outp[x:x+p_sz, y:y+p_sz] = gaussian_sample(noisy, patch, patches[inds], w, noise).reshape((p_sz, p_sz, 3))

    return outp


# im = plt.imread('Rec12_vis/texture13.jpg')/255.
im = (plt.imread('Rec12_vis/tokyo.jpg')[25:, 25:]/255.).astype(np.float32)

noise = (25/255)**2
noisy = im + np.sqrt(noise)*np.random.randn(*im.shape)
den = noisy.copy()

T = 2000
jump = 20
fig, ax = plt.subplots()
I = ax.imshow(np.clip(den, 0, 1))
ax.axis('off')
text = ax.text(.01, .01, f'0/{T}', color='r', verticalalignment='top', horizontalalignment='left')
pbar = tqdm(range(T//jump))


def init():
    ax.set_aspect(1)
    return I,


def update(frame):
    global I, den
    pbar.update(1)
    # print(frame+1, flush=True)
    # for col in I.collections: plt.gca().collections.remove(col)
    den = gibbs_denoise(noisy, den, noise, p_sz=8, its=jump, beta=.01, k=16, m=None)
    I = ax.imshow(np.clip(den, 0, 1))
    text.set_text(f'{jump*(frame + 1)}/{T}')
    return I,


ani = FuncAnimation(fig, update, frames=T//jump, init_func=init, blit=True,)
ani.save('denoise.gif', fps=(T//jump)//10)
plt.imsave('denoised.png', np.clip(den, 0, 1))
plt.imsave('noisy.png', np.clip(noisy, 0, 1))