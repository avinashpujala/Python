from sympy import *
import numpy as np
import scipy.optimize

w, x, y, θ = symbols('w x y θ')

def norm(v):
    (x, y) = v
    return sqrt(x**2 + y**2)

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

def pushAlong(x, y):
    scalar = dot(x, y) / dot(y, y)
    return (scalar * y[0], scalar * y[1])

def vec_minus(a, b):
    return (a[0] - b[0], a[1] - b[1])

def distance_point_line(x, y, θ):
    a = (x, y)
    b = (cos(θ), sin(θ))
    c = pushAlong(a, b)
    return norm(vec_minus(a, c))

def error_term_point(val_x, val_y, val_w):
    return val_w * distance_point_line(x, y, θ).subs('x', val_x).subs('y', val_y)

def find_orientation(xs, ys, ws):
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    error_terms = sum(map(error_term_point, xs, ys, ws))
    error_terms_f = lambdify(θ, error_terms)
    min_theta = scipy.optimize.minimize(error_terms_f, 0)
    return min_theta.x % np.pi

def weighted_centroid(xs, ys, ws):
    x = sum(xs * ws) / sum(ws)
    y = sum(ys * ws) / sum(ws)
    return (x, y)

def weighted_centroid_image(image):
    ys, xs = (image != np.nan).nonzero()
    ws = image[ys, xs]
    return weighted_centroid(xs, ys, ws)

def orientation_moments(blob):
    μ_ji = blob.weighted_moments_central
    cov = np.array([
        [μ_ji[2, 0], μ_ji[1, 1]],
        [μ_ji[1, 1], μ_ji[0, 2]]
    ]) / μ_ji[0, 0]

    μp_20 = μ_ji[2, 0] / μ_ji[0, 0]
    μp_02 = μ_ji[0, 2] / μ_ji[0, 0]
    μp_11 = μ_ji[1, 1] / μ_ji[0, 0]

    expr = (2 * μp_11) / (μp_20 - μp_02)
    θ_my = 0.5 * np.arctan(expr)

    if θ_my != θ_my:
        return np.pi / 4
    if (2 * μp_11) < 0:
        return ((-θ_my) % (np.pi / 2))
    else:
        return ((-θ_my) % (np.pi / 2)) - (np.pi / 2)
