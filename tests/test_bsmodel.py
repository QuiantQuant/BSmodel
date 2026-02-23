import pytest

from math import log, sqrt

from bsmodel.models import BSModel

@pytest.mark.parametrize(
    "inputs, px, delta, gamma, vega, theta, rho",
    [
        ([100.0, 100.0, 365, 0.200, 0.05, 0.00], 10.4506, 0.6368, 0.0188, 0.3752, -0.0176, 0.5323),
        ([150.0, 100.0, 365, 0.200, 0.05, 0.00], 54.9701, 0.9913, 0.0008, 0.0355, -0.0138, 0.9372),
        ([50.0, 100.0, 365, 0.200, 0.05, 0.00], 0.0024, 0.0009, 0.0003, 0.0016, -0.0000, 0.0004),
        ([100.0, 100.0, 1, 0.200, 0.05, 0.00], 0.4245, 0.5073, 0.3810, 0.0209, -0.2157, 0.0014),
        ([100.0, 100.0, 1825, 0.200, 0.05, 0.00], 29.1386, 0.7831, 0.0066, 0.6567, -0.0103, 2.4584),
        ([100.0, 100.0, 365, 0.800, 0.05, 0.00], 32.8210, 0.6781, 0.0045, 0.3585, -0.0441, 0.3499),
        ([100.0, 100.0, 365, 0.200, 0.00, 0.00], 7.9656, 0.5398, 0.0198, 0.3970, -0.0109, 0.4602),
        ([100.0, 100.0, 365, 0.200, 0.50, 0.00], 39.3780, 0.9953, 0.0007, 0.0136, -0.0828, 0.6016),
        ([100.0, 100.0, 365, 0.001, 0.05, 0.00], 4.8771, 1.0000, 0.0000, 0.0000, -0.0130, 0.9512),
        ([100.0, 100.0, 365, 0.200, 0.05, 0.03], 8.6525, 0.5621, 0.0190, 0.3795, -0.0123, 0.4756),
        ([100.0, 100.0, 365, 0.200, 0.05, 0.08], 6.1430, 0.4432, 0.0184, 0.3678, -0.0056, 0.3817),
    ],
)
class TestBSModelCall:
    @pytest.fixture(autouse=True)
    def setup(self, inputs, px, delta, gamma, vega, theta, rho):
        self.model = BSModel(*inputs)

        self.px = px
        self.delta = delta
        self.gamma = gamma
        self.vega = vega
        self.theta = theta
        self.rho = rho

    @pytest.mark.unit
    def test_price_call(self):
        assert self.model.price(option_type="call") == pytest.approx(self.px, abs=0.005)

    @pytest.mark.unit
    def test_delta_call(self):
        assert self.model.delta(option_type="call") == pytest.approx(self.delta, abs=0.005)

    @pytest.mark.unit
    def test_gamma_call(self):
        assert self.model.gamma(option_type="call") == pytest.approx(self.gamma, abs=0.005)

    @pytest.mark.unit
    def test_vega_call(self):
        assert self.model.vega(option_type="call") == pytest.approx(self.vega, abs=0.005)

    @pytest.mark.unit
    def test_theta_call(self):
        assert self.model.theta(option_type="call") == pytest.approx(self.theta, abs=0.005)

    @pytest.mark.unit
    def test_rho_call(self):
        assert self.model.rho(option_type="call") == pytest.approx(self.rho, abs=0.005)

    @pytest.mark.unit
    def test_call_put_parity(self):
        pass


@pytest.mark.unit
@pytest.mark.parametrize(
    "inputs",
    [
        ([100, 95, 0, 30, 10]),
        ([100, 105, -1, 30, 10]),
    ],
)
def test_d1_d2_edge_cases(inputs):
    model = BSModel(*inputs)

    model._calculate_d1_d2()

    assert model.d1 == 0
    assert model.d2 == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    "inputs",
    [
        ([100, 95, 365, 30, 10, 3]),
        ([100, 105, 365, 30, 10, 3]),
        ([100, 95, 5, 30, 10, 1]),
        ([100, 95, 365, 15, 10, 0]),
        ([100, 95, 365, 30, 5, 0]),
    ],
)
def test_d1_d2(inputs):
    model = BSModel(*inputs)

    spot = inputs[0]
    strike = inputs[1]
    expiration = inputs[2] / 365
    volatility = inputs[3]
    rate = inputs[4]
    dividend = inputs[5]
    model._calculate_d1_d2()

    expected_d1 = (log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * expiration) / (
        volatility * sqrt(expiration)
    )
    expected_d2 = expected_d1 - volatility * sqrt(expiration)

    assert model.d1 == pytest.approx(expected_d1)
    assert model.d2 == pytest.approx(expected_d2)
