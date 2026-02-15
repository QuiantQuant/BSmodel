from math import log, sqrt, exp
from scipy.stats import norm


class BSModel:
    def __init__(
        self,
        spot: float,
        strike: float,
        expiration: int,
        volatility: float,
        rate: float,
        dividend: float = 0,
    ):
        self.spt = spot
        self.stk = strike
        self.tte = expiration / 365
        self.vol = volatility / 100
        self.rfr = rate / 100
        self.dvd = dividend / 100

        # Pre-calculate d1 and d2 for reuse
        self._calculate_d1_d2()

    def _calculate_d1_d2(self):
        if self.tte <= 0:
            self.d1, self.d2 = 0, 0
            return

        self.d1 = (log(self.spt / self.stk) + (self.rfr - self.dvd + 0.5 * self.vol**2) * self.tte) / (
            self.vol * sqrt(self.tte)
        )
        self.d2 = self.d1 - self.vol * sqrt(self.tte)

    def price_call(self):
        return self.spt * exp(-self.dvd * self.tte) * norm.cdf(self.d1) - self.stk * exp(
            -self.rfr * self.tte
        ) * norm.cdf(self.d2)

    def price_put(self):
        return self.stk * exp(-self.rfr * self.tte) * norm.cdf(-self.d2) - self.spt * exp(
            -self.dvd * self.tte
        ) * norm.cdf(-self.d1)
