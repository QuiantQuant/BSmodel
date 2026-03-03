from math import log, sqrt, exp
from scipy.stats import norm


def precalc_common(S, K, t, v, r):
    return log(S / K) / (v * sqrt(t)) + r * sqrt(t) / v


def precalc_diff(S, K, t, v, r):
    return v * sqrt(t) / 2


def d1(S, K, t, v, r):
    return precalc_common(S, K, t, v, r) + precalc_diff(S, K, t, v, r)


def d2(S, K, t, v, r):
    return precalc_common(S, K, t, v, r) - precalc_diff(S, K, t, v, r)


def price_call(S, K, t, v, r):
    return S * norm.cdf(d1(S, K, t, v, r)) - K * exp(-r * t) * norm.cdf(d2(S, K, t, v, r))


def delta_call(S, K, t, v, r):
    return norm.cdf(d1(S, K, t, v, r))


def gamma_call(S, K, t, v, r):
    return norm.pdf(d1(S, K, t, v, r)) / (S * v * sqrt(t))


def vega_call(S, K, t, v, r):
    return S * norm.pdf(d1(S, K, t, v, r)) * sqrt(t) / 100


def theta_call(S, K, t, v, r):
    return (
        -S * norm.pdf(d1(S, K, t, v, r)) * v / (2 * sqrt(t)) - K * r * exp(-r * t) * norm.cdf(d2(S, K, t, v, r))
    ) / 365


def rho_call(S, K, t, v, r):
    return K * t * exp(-r * t) * norm.cdf(d2(S, K, t, v, r)) / 100
