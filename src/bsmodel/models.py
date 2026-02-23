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
        self.spt: float = spot
        self.stk: float = strike
        self.tte: float = expiration / 365
        self.vol: float = volatility
        self.rfr: float = rate
        self.dvd: float = dividend

        # Pre-calculate d1 and d2 for reuse
        self._calculate_d1_d2()

    def _calculate_d1_d2(self) -> None:
        if self.tte <= 0:
            self.d1, self.d2 = 0, 0
            return

        self.d1: float = (log(self.spt / self.stk) + (self.rfr - self.dvd + 0.5 * self.vol**2) * self.tte) / (
            self.vol * sqrt(self.tte)
        )
        self.d2: float = self.d1 - self.vol * sqrt(self.tte)

    def price(self, option_type:str = "call") -> float:
        if option_type == "call":
            return self.spt * exp(-self.dvd * self.tte) * norm.cdf(self.d1) - self.stk * exp(
                -self.rfr * self.tte
                ) * norm.cdf(self.d2)
        else: # Put
            return self.stk * exp(-self.rfr * self.tte) * norm.cdf(-self.d2) - self.spt * exp(
                -self.dvd * self.tte
            ) * norm.cdf(-self.d1)

    def delta(self, option_type:str = "call") -> float:
        return exp(-self.dvd * self.tte) * norm.cdf(self.d1)

    def gamma(self, option_type:str = "call") -> float:
        return exp(-self.dvd * self.tte) * norm.pdf(self.d1) / (self.spt * self.vol * sqrt(self.tte))

    def vega(self, option_type:str = "call") -> float:
        return self.spt * exp(-self.dvd * self.tte) * norm.pdf(self.d1) * sqrt(self.tte) / 100

    def theta(self, option_type:str = "call") -> float:
        if self.tte <= 0:
            return 0.0

        return (-(self.spt * exp(-self.dvd*self.tte) * norm.pdf(self.d1) * self.vol) / (2 * sqrt(self.tte))
         - self.rfr * self.stk * exp(-self.rfr*self.tte) * norm.cdf(self.d2)
         + self.dvd * self.spt * exp(-self.dvd*self.tte) * norm.cdf(self.d1)) / 365

    def rho(self, option_type:str = "call") -> float:
        return self.stk * self.tte * exp(-self.rfr * self.tte) * norm.cdf(self.d2) / 100
