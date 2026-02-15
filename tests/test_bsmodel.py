from bsmodel.models import BSModel
from math import log, sqrt
import pytest


class TestBSModel:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "inputs, expected_result",
        [
            ([100, 95, 365, 30, 10, 3], 17.32),
            ([100, 105, 365, 30, 10, 3], 12.50),
            ([100, 95, 5, 30, 10, 3], 5.19),
            ([100, 95, 365, 15, 10, 3], 12.70),
            ([100, 95, 365, 30, 5, 3], 14.82),
            ([100, 95, 365, 30, 10, 1], 18.74),
        ],
    )
    def test_price_call(self, inputs, expected_result):
        model = BSModel(*inputs)
        call_price = model.price_call()

        assert call_price == pytest.approx(expected_result, abs=0.005)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "inputs, expected_result",
        [
            ([100, 95, 365, 30, 10], 19.47),
            ([100, 105, 365, 30, 10], 14.29),
            ([100, 95, 5, 30, 10], 5.23),
            ([100, 95, 365, 15, 10], 15.18),
            ([100, 95, 365, 30, 5], 16.80),
        ],
    )
    def test_price_call_zero_dividend(self, inputs, expected_result):
        model = BSModel(*inputs)
        call_price = model.price_call()

        assert call_price == pytest.approx(expected_result, abs=0.005)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "inputs, expected_result",
        [
            ([100, 95, 365, 30, 10, 3], 6.24),
            ([100, 105, 365, 30, 10, 3], 10.46),
            ([100, 95, 30, 30, 10, 3], 1.27),
            ([100, 95, 365, 15, 10, 3], 1.62),
            ([100, 95, 365, 30, 5, 3], 8.15),
            ([100, 95, 365, 30, 10, 1], 5.69),
        ],
    )
    def test_price_put(self, inputs, expected_result):
        model = BSModel(*inputs)
        put_price = model.price_put()

        assert put_price == pytest.approx(expected_result, abs=0.005)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "inputs, expected_result",
        [
            ([100, 95, 365, 30, 10], 5.43),
            ([100, 105, 365, 30, 10], 9.30),
            ([100, 95, 5, 30, 10], 0.10),
            ([100, 95, 365, 15, 10], 1.14),
            ([100, 95, 365, 30, 5], 7.17),
        ],
    )
    def test_price_put_zero_dividend(self, inputs, expected_result):
        model = BSModel(*inputs)
        put_price = model.price_put()

        assert put_price == pytest.approx(expected_result, abs=0.005)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "inputs",
        [
            ([100, 95, 0, 30, 10]),
            ([100, 105, -1, 30, 10]),
        ],
    )
    def test_d1_d2_edge_cases(self, inputs):
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
    def test_d1_d2(self, inputs):
        model = BSModel(*inputs)

        spot = inputs[0]
        strike = inputs[1]
        expiration = inputs[2] / 365
        volatility = inputs[3] / 100
        rate = inputs[4] / 100
        dividend = inputs[5] / 100
        model._calculate_d1_d2()

        expected_d1 = (log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * expiration) / (
            volatility * sqrt(expiration)
        )
        expected_d2 = expected_d1 - volatility * sqrt(expiration)

        assert model.d1 == pytest.approx(expected_d1)
        assert model.d2 == pytest.approx(expected_d2)
