"""Discrete parallel-beam tomography: system matrix, phantoms, reconstruction
algorithms (FBP, LSQR, Tikhonov, TV-regularised primal-dual) and error measures."""

import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from PIL import Image, ImageDraw, ImageFont


# ------------------------------------------------------------------ phantoms
def _font(size):
    """DejaVu Sans as shipped with matplotlib, so that the phantom is identical on every platform."""
    import matplotlib
    path = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
    return ImageFont.truetype(path, size=size)


def digits_phantom(n=100, text="0038"):
    """White digits on a black n x n background, centred horizontally."""
    img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(img)
    font = _font(int(0.16 * n))
    w = draw.textlength(text, font=font)
    draw.text(((n - w) / 2, 0.38 * n), text, fill=255, font=font)
    return np.asarray(img, float) / 255.0


def shepp_logan(n=100):
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize
    return np.clip(resize(shepp_logan_phantom(), (n, n), anti_aliasing=True), 0, None)


# ------------------------------------------------------------------ forward model
def system_matrix(n, angles_deg, n_det=None):
    """Pixel-driven parallel-beam projector.

    Pixel (i, j) has centre (x, y) relative to the image centre; at angle theta
    its mass is split linearly between the two detector bins adjacent to
    s = x cos(theta) + y sin(theta).  Row index = angle * n_det + bin.
    Returns a CSR matrix of shape (len(angles) * n_det, n * n) and n_det."""
    if n_det is None:
        n_det = int(np.ceil(np.sqrt(2) * n)) + 1
    c = (n - 1) / 2.0
    jj, ii = np.meshgrid(np.arange(n), np.arange(n))
    x = (jj - c).ravel()
    y = (c - ii).ravel()
    cols = np.arange(n * n)
    rows, vals, cc = [], [], []
    for a, th in enumerate(np.deg2rad(angles_deg)):
        s = x * np.cos(th) + y * np.sin(th) + (n_det - 1) / 2.0
        k = np.floor(s).astype(int)
        f = s - k
        for kk, w in ((k, 1 - f), (k + 1, f)):
            ok = (kk >= 0) & (kk < n_det)
            rows.append(a * n_det + kk[ok]); vals.append(w[ok]); cc.append(cols[ok])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cc))),
                      shape=(len(angles_deg) * n_det, n * n))
    return A, n_det


def angles(m, span=180.0):
    return np.arange(m) * span / m


class Projector:
    """A together with a CSR copy of its transpose (faster adjoint products)."""

    def __init__(self, A):
        self.A = A
        self.AT = A.T.tocsr()
        self.shape = A.shape

    def __matmul__(self, x):
        return self.A @ x

    def adj(self, y):
        return self.AT @ y


# ------------------------------------------------------------------ filtered back-projection
def ramp_filter(n_det, window="ram-lak"):
    """Frequency response of the band-limited ramp filter (Ram-Lak spatial kernel,
    unit detector spacing) on a zero-padded FFT grid, optionally apodised."""
    pad = int(2 ** np.ceil(np.log2(2 * n_det)))
    k = np.arange(-(n_det - 1), n_det)
    h = np.zeros(k.size)
    h[k == 0] = 0.25
    odd = (k % 2) == 1
    h[odd] = -1.0 / (np.pi * k[odd]) ** 2
    hc = np.zeros(pad)
    hc[k % pad] = h                       # circular embedding of the symmetric kernel
    H = np.real(np.fft.fft(hc))
    f = np.fft.fftfreq(pad)               # cycles per detector bin, |f| <= 1/2
    if window == "shepp-logan":
        H *= np.sinc(f)
    elif window == "hann":
        H *= 0.5 * (1 + np.cos(2 * np.pi * f))
    elif window != "ram-lak":
        raise ValueError(window)
    return H, pad


def fbp(P, b, n, n_angles, n_det, window="ram-lak", span=180.0, clip=True):
    """x(p) = (span/m) sum_j q_j(p . theta_j) with q_j = h * p_j, evaluated by the
    adjoint of the same discrete projector (linear interpolation in s)."""
    H, pad = ramp_filter(n_det, window)
    sino = b.reshape(n_angles, n_det)
    q = np.real(np.fft.ifft(np.fft.fft(sino, pad, axis=1) * H, axis=1))[:, :n_det]
    x = (np.deg2rad(span) / n_angles) * P.adj(q.ravel())
    if clip:
        x = np.clip(x, 0, None)
    return x.reshape(n, n)


# ------------------------------------------------------------------ Krylov and Tikhonov
def lsqr_path(P, b, kmax, record=(), x_true=None):
    """LSQR (Golub-Kahan bidiagonalisation, Paige and Saunders 1982) started at x0 = 0.

    Returns the iterates with index in `record`, the residual norms ||A x_k - b||
    for k = 1..kmax and, if x_true is given, the relative errors for k = 1..kmax."""
    record = set(record)
    beta = np.linalg.norm(b); u = b / beta
    v = P.adj(u); alpha = np.linalg.norm(v); v = v / alpha
    w = v.copy(); x = np.zeros(P.shape[1])
    phibar, rhobar = beta, alpha
    res, err, out = np.empty(kmax), np.empty(kmax), {}
    nt = np.linalg.norm(x_true) if x_true is not None else 1.0
    for k in range(1, kmax + 1):
        u = P @ v - alpha * u; beta = np.linalg.norm(u); u /= beta
        v = P.adj(u) - beta * v; alpha = np.linalg.norm(v); v /= alpha
        rho = np.hypot(rhobar, beta)
        c, s = rhobar / rho, beta / rho
        theta = s * alpha; rhobar = -c * alpha
        phi = c * phibar; phibar = s * phibar
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w
        res[k - 1] = abs(phibar)
        if x_true is not None:
            err[k - 1] = np.linalg.norm(x - x_true.ravel()) / nt
        if k in record:
            out[k] = x.copy()
    return out, res, (err if x_true is not None else None)


def tikhonov(A, b, lam, iters=500):
    """argmin ||A x - b||^2 + lam^2 ||x||^2 by damped LSQR."""
    return lsqr(A, b, damp=lam, iter_lim=iters, atol=1e-10, btol=1e-10)[0]


# ------------------------------------------------------------------ total variation
def grad(u):
    gx = np.zeros_like(u); gy = np.zeros_like(u)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return gx, gy


def div(px, py):
    """Negative adjoint of grad."""
    d = np.zeros_like(px)
    d[:, 0] = px[:, 0]; d[:, 1:-1] = px[:, 1:-1] - px[:, :-2]; d[:, -1] = -px[:, -2]
    e = np.zeros_like(py)
    e[0, :] = py[0, :]; e[1:-1, :] = py[1:-1, :] - py[:-2, :]; e[-1, :] = -py[-2, :]
    return d + e


def tv(u):
    gx, gy = grad(u)
    return float(np.sum(np.sqrt(gx ** 2 + gy ** 2)))


def op_norm(P, iters=50, seed=0):
    v = np.random.default_rng(seed).standard_normal(P.shape[1])
    for _ in range(iters):
        v = P.adj(P @ v)
        v /= np.linalg.norm(v)
    return float(np.sqrt(np.linalg.norm(P.adj(P @ v))))


def tv_pdhg(P, b, n, lam, iters=600, nonneg=True, LA=None, history=None, x_true=None):
    """min_{x >= 0} 1/2 ||A x - b||^2 + lam TV(x)   (isotropic TV)
    by the primal-dual hybrid gradient method of Chambolle and Pock (2011)
    with K = [A; lam grad], ||grad||^2 <= 8 and tau = sigma = 0.99 / ||K||.
    If `history` is a list, (iteration, objective, relative error) is appended every 10 iterations."""
    if LA is None:
        LA = op_norm(P)
    L = np.sqrt(LA ** 2 + 8 * lam ** 2)
    tau = sigma = 0.99 / L
    x = np.zeros((n, n)); xb = x.copy()
    y = np.zeros(P.shape[0]); px = np.zeros((n, n)); py = np.zeros((n, n))
    for it in range(1, iters + 1):
        # dual step for the data term: prox of f*(y) with f = 1/2 ||. - b||^2
        y = (y + sigma * (P @ xb.ravel() - b)) / (1 + sigma)
        # dual step for TV: pointwise projection onto the unit l2 ball
        gx, gy = grad(xb)
        px += sigma * lam * gx; py += sigma * lam * gy
        nrm = np.maximum(1.0, np.sqrt(px ** 2 + py ** 2))
        px /= nrm; py /= nrm
        x_old = x
        x = x - tau * (P.adj(y).reshape(n, n) - lam * div(px, py))
        if nonneg:
            x = np.maximum(x, 0)
        xb = 2 * x - x_old
        if history is not None and it % 10 == 0:
            obj = 0.5 * np.linalg.norm(P @ x.ravel() - b) ** 2 + lam * tv(x)
            e = np.linalg.norm(x - x_true) / np.linalg.norm(x_true) if x_true is not None else np.nan
            history.append((it, obj, e))
    return x


# ------------------------------------------------------------------ metrics
def metrics(x, x_true):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    x = np.asarray(x).reshape(x_true.shape)
    return {"rel_err": float(np.linalg.norm(x - x_true) / np.linalg.norm(x_true)),
            "psnr": float(peak_signal_noise_ratio(x_true, np.clip(x, 0, 1), data_range=1.0)),
            "ssim": float(structural_similarity(x_true, np.clip(x, 0, 1), data_range=1.0))}
