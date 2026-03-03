import pytest

import bsmodel.bs_formulas as bsf


@pytest.mark.parametrize(
    "inputs, price, delta, gamma, vega, theta, rho",
    [
        ([100.0, 100.0, 1, 0.200, 0.05], 10.45, 0.63683, 0.01876, 0.37524, -0.01757, 0.53232),
        ([150.0, 100.0, 1, 0.200, 0.05], 54.97, 0.99128, 0.00079, 0.03546, -0.01381, 0.93722),
        ([50.00, 100.0, 1, 0.200, 0.05], 0.000, 0.00092, 0.00031, 0.00156, -0.00005, 0.00043),
        ([100.0, 100.0, 1 / 365, 0.200, 0.05], 0.420, 0.50731, 0.38103, 0.02088, -0.21567, 0.00138),
        ([100.0, 100.0, 1825 / 365, 0.200, 0.05], 29.14, 0.78308, 0.00657, 0.65674, -0.01033, 2.45845),
        ([100.0, 100.0, 1, 0.800, 0.05], 32.82, 0.67814, 0.00448, 0.35848, -0.04408, 0.34993),
        ([100.0, 100.0, 1, 0.200, 0.00], 7.970, 0.53983, 0.01985, 0.39695, -0.01088, 0.46017),
        ([100.0, 100.0, 1, 0.200, 0.50], 39.38, 0.99534, 0.00068, 0.01358, -0.08278, 0.60156),
        ([100.0, 100.0, 1, 0.001, 0.05], 4.880, 1.00000, 0.00000, 0.00000, -0.01303, 0.95123),
    ],
)
class TestBSModelCall:
    @pytest.fixture(autouse=True)
    def setup(self, inputs, price, delta, gamma, vega, theta, rho):
        self.inputs = inputs

        self.price = price
        self.delta = delta
        self.gamma = gamma
        self.vega = vega
        self.theta = theta
        self.rho = rho

    @pytest.mark.unit
    def test_price_call(self):
        assert bsf.price_call(*self.inputs) == pytest.approx(self.price, abs=0.05)

    @pytest.mark.unit
    def test_delta_call(self):
        assert bsf.delta_call(*self.inputs) == pytest.approx(self.delta, abs=0.00005)

    @pytest.mark.unit
    def test_gamma_call(self):
        assert bsf.gamma_call(*self.inputs) == pytest.approx(self.gamma, abs=0.00005)

    @pytest.mark.unit
    def test_vega_call(self):
        assert bsf.vega_call(*self.inputs) == pytest.approx(self.vega, abs=0.00005)

    @pytest.mark.unit
    def test_theta_call(self):
        assert bsf.theta_call(*self.inputs) == pytest.approx(self.theta, abs=0.00005)

    @pytest.mark.unit
    def test_rho_call(self):
        assert bsf.rho_call(*self.inputs) == pytest.approx(self.rho, abs=0.00005)
